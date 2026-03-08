import { factories } from '@strapi/strapi';

const UID = 'api::tag.tag';
const POST_UID = 'api::post.post';

function toPagination(query: any) {
  const page = Math.max(1, Number(query?.pagination?.page ?? 1) || 1);
  const pageSize = Math.min(100, Math.max(1, Number(query?.pagination?.pageSize ?? 10) || 10));
  return { page, pageSize };
}

function normalizeAliases(value: unknown) {
  const source = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(',')
      : [];

  const dedup = new Map<string, string>();
  for (const item of source) {
    const text = String(item ?? '').trim();
    if (!text) {
      continue;
    }
    const key = text.toLowerCase();
    if (!dedup.has(key)) {
      dedup.set(key, text);
    }
  }
  return Array.from(dedup.values());
}

function normalizeTagPayload(payload: any) {
  const data = { ...(payload ?? {}) };
  if ('aliases' in data) {
    data.aliases = normalizeAliases(data.aliases);
  }
  return data;
}

async function findTagByDocumentId(strapi: any, documentId: string) {
  return strapi.documents(UID).findOne({
    documentId,
    populate: ['posts'],
  });
}

async function findDocumentPostIdsForTag(strapi: any, tagDocumentId: string) {
  const pageSize = 200;
  const postDocumentIds = new Set<string>();
  const postDocuments = strapi.documents(POST_UID) as any;

  for (const status of ['draft', 'published'] as const) {
    let page = 1;
    while (true) {
      const batch = (await postDocuments.findMany({
        fields: ['documentId'],
        filters: {
          tags: {
            documentId: { $eq: tagDocumentId },
          },
        },
        pagination: { page, pageSize },
        status,
      })) as Array<{ documentId?: string }>;

      if (!Array.isArray(batch) || batch.length === 0) {
        break;
      }

      for (const item of batch) {
        const documentId = String(item?.documentId ?? '').trim();
        if (documentId) {
          postDocumentIds.add(documentId);
        }
      }

      if (batch.length < pageSize) {
        break;
      }
      page += 1;
    }
  }

  return Array.from(postDocumentIds);
}

export default factories.createCoreService(UID, ({ strapi }) => ({
  async listForAdmin(query: any) {
    const { page, pageSize } = toPagination(query);
    const sort = query?.sort ?? 'updatedAt:desc';
    const keyword = String(query?.q ?? '').trim();
    const keywordFilters = keyword
      ? {
          $or: [
            { name: { $containsi: keyword } },
            { slug: { $containsi: keyword } },
          ],
        }
      : undefined;
    const filters = query?.filters
      ? keywordFilters
        ? { $and: [query.filters, keywordFilters] }
        : query.filters
      : keywordFilters;

    const [rows, total] = await Promise.all([
      strapi.documents(UID).findMany({
        sort,
        filters,
        populate: query?.populate,
        pagination: { page: 1, pageSize: 1000 },
      }),
      strapi.documents(UID).count({ filters }),
    ]);

    const start = Math.max(0, (page - 1) * pageSize);
    const data = (rows as any[]).slice(start, start + pageSize);

    return {
      data,
      meta: {
        pagination: {
          page,
          pageSize,
          pageCount: Math.max(1, Math.ceil(total / pageSize)),
          total,
        },
      },
    };
  },

  async findOneForAdmin(documentId: string, query: any) {
    return strapi.documents(UID).findOne({
      documentId,
      populate: query?.populate,
      fields: query?.fields,
      locale: query?.locale,
    });
  },

  async createForAdmin(payload: any) {
    const data = normalizeTagPayload(payload);
    return strapi.documents(UID).create({
      data,
    });
  },

  async updateForAdmin(documentId: string, payload: any) {
    const data = normalizeTagPayload(payload);
    return strapi.documents(UID).update({
      documentId,
      data,
    });
  },

  async deleteForAdmin(documentId: string) {
    return strapi.documents(UID).delete({ documentId });
  },

  async mergeForAdmin(sourceDocumentId: string, targetDocumentId: string) {
    if (sourceDocumentId === targetDocumentId) {
      throw new Error('Cannot merge a tag into itself');
    }

    const sourceTag = await findTagByDocumentId(strapi, sourceDocumentId);
    const targetTag = await findTagByDocumentId(strapi, targetDocumentId);

    if (!sourceTag || !targetTag) {
      throw new Error('Tag not found');
    }

    const postDocumentIds = await findDocumentPostIdsForTag(strapi, sourceDocumentId);
    const postDocuments = strapi.documents(POST_UID) as any;

    for (const postDocumentId of postDocumentIds) {
      const postEntry =
        (await postDocuments.findOne({
          documentId: postDocumentId,
          populate: ['tags'],
          status: 'draft',
        }));

      if (!postEntry) {
        continue;
      }

      const nextTagIds = Array.from(
        new Set(
          (postEntry.tags ?? [])
            .map((tag: { documentId?: string }) => String(tag?.documentId ?? '').trim())
            .filter((tagId: string) => Boolean(tagId) && tagId !== sourceDocumentId)
            .concat(targetDocumentId),
        ),
      );

      await postDocuments.update({
        documentId: postDocumentId,
        data: { tags: nextTagIds },
      });
    }

    const mergedAliases = normalizeAliases([
      ...(Array.isArray(targetTag.aliases) ? targetTag.aliases : []),
      ...(Array.isArray(sourceTag.aliases) ? sourceTag.aliases : []),
      sourceTag.name,
    ]);

    const updatedTarget = await strapi.documents(UID).update({
      documentId: targetDocumentId,
      data: {
        aliases: mergedAliases,
      },
    });

    await strapi.documents(UID).delete({ documentId: sourceDocumentId });

    return {
      target: updatedTarget,
      mergedPostCount: postDocumentIds.length,
    };
  },
}));
