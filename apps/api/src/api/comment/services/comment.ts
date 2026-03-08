import { factories } from '@strapi/strapi';

const UID = 'api::comment.comment';
const POST_UID = 'api::post.post';
const USER_UID = 'plugin::users-permissions.user';

function toPagination(query: any) {
  const page = Math.max(1, Number(query?.pagination?.page ?? 1) || 1);
  const pageSize = Math.min(100, Math.max(1, Number(query?.pagination?.pageSize ?? 10) || 10));
  return { page, pageSize };
}

async function ensureValidParent(
  strapi: any,
  parentDocumentId: string | null,
  targetType: string,
  targetDocumentId: string,
  currentDocumentId?: string,
) {
  if (!parentDocumentId) {
    return null;
  }

  const parent = await strapi.documents(UID).findOne({
    documentId: parentDocumentId,
    fields: ['documentId', 'targetType', 'targetDocumentId'],
  });

  if (!parent) {
    throw new Error('Parent comment not found');
  }

  if (currentDocumentId && parent.documentId === currentDocumentId) {
    throw new Error('Comment cannot reply to itself');
  }

  if (parent.targetType !== targetType || parent.targetDocumentId !== targetDocumentId) {
    throw new Error('Parent comment must belong to the same target');
  }

  return parent.documentId as string;
}

async function normalizeCommentPayload(strapi: any, payload: any, currentDocumentId?: string) {
  const data = { ...(payload ?? {}) };
  const targetType = String(data.targetType ?? '').trim();
  const targetDocumentId = String(data.targetDocumentId ?? '').trim();

  if (!targetType || !targetDocumentId) {
    throw new Error('targetType and targetDocumentId are required');
  }

  const rawParent = String(data.parent ?? data.parentDocumentId ?? '').trim();
  data.parent = await ensureValidParent(
    strapi,
    rawParent || null,
    targetType,
    targetDocumentId,
    currentDocumentId,
  );
  delete data.parentDocumentId;

  return data;
}

function normalizeIdentity(value: unknown) {
  return String(value ?? '').trim().toLowerCase();
}

async function enrichCommentsWithAvatar(strapi: any, rows: any[]) {
  if (!rows.length) {
    return rows;
  }

  const emails = Array.from(
    new Set(
      rows
        .map((row) => normalizeIdentity(row?.authorEmail))
        .filter(Boolean),
    ),
  );
  const usernames = Array.from(
    new Set(
      rows
        .map((row) => normalizeIdentity(row?.authorName))
        .filter(Boolean),
    ),
  );

  if (!emails.length && !usernames.length) {
    return rows;
  }

  const orFilters: any[] = [];
  emails.forEach((email) => {
    orFilters.push({ email: { $eqi: email } });
  });
  usernames.forEach((username) => {
    orFilters.push({ username: { $eqi: username } });
  });

  const users = (await strapi.query(USER_UID).findMany({
    where: { $or: orFilters },
    select: ['id', 'username', 'email'],
    populate: {
      avatar: {
        select: ['url'],
      },
    },
  })) as Array<{
    username?: string | null;
    email?: string | null;
    avatar?: { url?: string | null } | null;
  }>;

  const avatarByEmail = new Map<string, string>();
  const avatarByUsername = new Map<string, string>();

  for (const user of users ?? []) {
    const avatarUrl = user?.avatar?.url;
    if (!avatarUrl) {
      continue;
    }
    const emailKey = normalizeIdentity(user.email);
    if (emailKey) {
      avatarByEmail.set(emailKey, avatarUrl);
    }
    const usernameKey = normalizeIdentity(user.username);
    if (usernameKey) {
      avatarByUsername.set(usernameKey, avatarUrl);
    }
  }

  return rows.map((row) => {
    const avatarUrl =
      avatarByEmail.get(normalizeIdentity(row?.authorEmail)) ??
      avatarByUsername.get(normalizeIdentity(row?.authorName)) ??
      null;
    return {
      ...row,
      authorAvatarUrl: avatarUrl,
    };
  });
}

export default factories.createCoreService(UID, ({ strapi }) => ({
  async listPublic(query: any) {
    const { page, pageSize } = toPagination(query);
    const sort = query?.sort ?? 'createdAt:asc';
    const filters = query?.filters;

    const [data, total] = await Promise.all([
      strapi.documents(UID).findMany({
        sort,
        filters,
        populate: query?.populate ?? {
          parent: {
            fields: ['documentId', 'authorName', 'targetType', 'targetDocumentId'],
          },
        },
        pagination: { page, pageSize },
      }),
      strapi.documents(UID).count({ filters }),
    ]);

    const enriched = await enrichCommentsWithAvatar(strapi, data as any[]);

    return {
      data: enriched,
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

  async listForAdmin(query: any) {
    const { page, pageSize: requestedPageSize } = toPagination(query);
    const pageSize = Math.min(10, requestedPageSize);
    const sort = query?.sort ?? 'updatedAt:desc';
    const keyword = String(query?.q ?? '').trim();
    const keywordFilters = keyword
      ? {
          $or: [
            { authorName: { $containsi: keyword } },
            { authorEmail: { $containsi: keyword } },
            { content: { $containsi: keyword } },
            { targetDocumentId: { $containsi: keyword } },
          ],
        }
      : undefined;

    const filters = query?.filters
      ? keywordFilters
        ? { $and: [query.filters, keywordFilters] }
        : query.filters
      : keywordFilters;

    const [data, total] = await Promise.all([
      strapi.documents(UID).findMany({
        sort,
        filters,
        populate: {
          parent: {
            fields: ['documentId', 'authorName', 'targetType', 'targetDocumentId'],
          },
        },
        pagination: { page, pageSize },
      }),
      strapi.documents(UID).count({ filters }),
    ]);

    const postIds = Array.from(
      new Set(
        (data as any[])
          .filter((item) => item.targetType === 'post' && item.targetDocumentId)
          .map((item) => item.targetDocumentId),
      ),
    );

    const postMap = new Map<string, string>();
    if (postIds.length > 0) {
      const posts = (await strapi.documents(POST_UID).findMany({
        filters: { documentId: { $in: postIds } },
        fields: ['title'],
        pagination: { page: 1, pageSize: 1000 },
      })) as Array<{ documentId: string; title?: string }>;

      for (const post of posts) {
        postMap.set(post.documentId, post.title ?? post.documentId);
      }
    }

    const enriched = (data as any[]).map((item) => ({
      ...item,
      targetTitle:
        item.targetType === 'post'
          ? postMap.get(item.targetDocumentId) ?? item.targetDocumentId
          : item.targetDocumentId,
    }));

    return {
      data: enriched,
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
      populate: query?.populate ?? {
        parent: {
          fields: ['documentId', 'authorName', 'targetType', 'targetDocumentId'],
        },
      },
      fields: query?.fields,
      locale: query?.locale,
    });
  },

  async findOnePublic(documentId: string, query: any) {
    const data = await strapi.documents(UID).findOne({
      documentId,
      populate: query?.populate ?? {
        parent: {
          fields: ['documentId', 'authorName', 'targetType', 'targetDocumentId'],
        },
      },
      fields: query?.fields,
      locale: query?.locale,
    });
    if (!data) {
      return data;
    }
    const [enriched] = await enrichCommentsWithAvatar(strapi, [data]);
    return enriched;
  },

  async createPublic(payload: any) {
    const data = await normalizeCommentPayload(strapi, payload);
    const created = await strapi.documents(UID).create({ data });
    const [enriched] = await enrichCommentsWithAvatar(strapi, [created]);
    return enriched;
  },

  async createForAdmin(payload: any) {
    const data = await normalizeCommentPayload(strapi, payload);
    const created = await strapi.documents(UID).create({ data });
    const [enriched] = await enrichCommentsWithAvatar(strapi, [created]);
    return enriched;
  },

  async updatePublic(documentId: string, payload: any) {
    const data = await normalizeCommentPayload(strapi, payload, documentId);
    const updated = await strapi.documents(UID).update({ documentId, data });
    const [enriched] = await enrichCommentsWithAvatar(strapi, [updated]);
    return enriched;
  },

  async updateForAdmin(documentId: string, payload: any) {
    const data = await normalizeCommentPayload(strapi, payload, documentId);
    const updated = await strapi.documents(UID).update({ documentId, data });
    const [enriched] = await enrichCommentsWithAvatar(strapi, [updated]);
    return enriched;
  },

  async deletePublic(documentId: string) {
    return strapi.documents(UID).delete({ documentId });
  },

  async deleteForAdmin(documentId: string) {
    return strapi.documents(UID).delete({ documentId });
  },
}));
