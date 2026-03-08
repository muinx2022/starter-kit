import {
  checkAiProviderConnection,
  getAiAutomationSettings,
  runCommentAutomation,
  runContentAutomation,
  testCommentAutomation,
  testContentAutomation,
  updateAiAutomationSettings,
} from '../../../automation/ai-automation';

declare const strapi: any;

function readDocumentId(ctx: any) {
  const documentId = String(ctx.params?.documentId ?? '').trim();
  if (!documentId) {
    ctx.badRequest('documentId is required');
    return null;
  }
  return documentId;
}

function readUserId(ctx: any) {
  const id = Number(ctx.params?.id);
  if (!Number.isFinite(id)) {
    ctx.badRequest('Invalid user id');
    return null;
  }
  return id;
}

export default {
  async listPosts(ctx: any) {
    const service = strapi.service('api::post.post') as any;
    ctx.body = await service.listForAdmin(ctx.query);
  },

  async findPost(ctx: any) {
    const service = strapi.service('api::post.post') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.findOneForAdmin(documentId, ctx.query);
    if (!data) {
      return ctx.notFound('Post not found');
    }

    ctx.body = { data };
  },

  async createPost(ctx: any) {
    const service = strapi.service('api::post.post') as any;
    const data = await service.createForAdmin(ctx.request.body?.data);
    ctx.body = { data };
  },

  async updatePost(ctx: any) {
    const service = strapi.service('api::post.post') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.updateForAdmin(documentId, ctx.request.body?.data);
    ctx.body = { data };
  },

  async deletePost(ctx: any) {
    const service = strapi.service('api::post.post') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    await service.deleteForAdmin(documentId);
    ctx.body = { data: { documentId } };
  },

  async publishPost(ctx: any) {
    const service = strapi.service('api::post.post') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.publishForAdmin(documentId);
    ctx.body = { data };
  },

  async unpublishPost(ctx: any) {
    const service = strapi.service('api::post.post') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.unpublishForAdmin(documentId);
    ctx.body = { data };
  },

  async listPages(ctx: any) {
    const service = strapi.service('api::page.page') as any;
    ctx.body = await service.listForAdmin(ctx.query);
  },

  async findPage(ctx: any) {
    const service = strapi.service('api::page.page') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.findOneForAdmin(documentId, ctx.query);
    if (!data) {
      return ctx.notFound('Page not found');
    }

    ctx.body = { data };
  },

  async createPage(ctx: any) {
    const service = strapi.service('api::page.page') as any;
    const data = await service.createForAdmin(ctx.request.body?.data);
    ctx.body = { data };
  },

  async updatePage(ctx: any) {
    const service = strapi.service('api::page.page') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.updateForAdmin(documentId, ctx.request.body?.data);
    ctx.body = { data };
  },

  async deletePage(ctx: any) {
    const service = strapi.service('api::page.page') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    await service.deleteForAdmin(documentId);
    ctx.body = { data: { documentId } };
  },

  async listCategories(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    ctx.body = await service.listForAdmin(ctx.query);
  },

  async findCategory(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.findOneForAdmin(documentId, ctx.query);
    if (!data) {
      return ctx.notFound('Category not found');
    }

    ctx.body = { data };
  },

  async createCategory(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    const data = await service.createForAdmin(ctx.request.body?.data);
    ctx.body = { data };
  },

  async updateCategory(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.updateForAdmin(documentId, ctx.request.body?.data);
    ctx.body = { data };
  },

  async deleteCategory(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    await service.deleteForAdmin(documentId);
    ctx.body = { data: { documentId } };
  },

  async publishCategory(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.publishForAdmin(documentId);
    ctx.body = { data };
  },

  async unpublishCategory(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.unpublishForAdmin(documentId);
    ctx.body = { data };
  },

  async reorderCategories(ctx: any) {
    const service = strapi.service('api::category.category') as any;
    const rawDraggedId = ctx.request.body?.draggedId;
    const rawTargetId = ctx.request.body?.targetId;
    const draggedId = Number(rawDraggedId);
    const hasTargetId = !(rawTargetId === null || rawTargetId === undefined || rawTargetId === '');
    const targetId = hasTargetId ? Number(rawTargetId) : null;
    const position = String(ctx.request.body?.position ?? '');

    if (!Number.isFinite(draggedId)) {
      return ctx.badRequest('draggedId is required');
    }

    if (!['child', 'after', 'root'].includes(position)) {
      return ctx.badRequest("position must be 'child', 'after' or 'root'");
    }

    if (position !== 'root' && !Number.isFinite(targetId as number)) {
      return ctx.badRequest('targetId is required for child/after');
    }

    if (Number.isFinite(targetId as number) && draggedId === targetId) {
      return ctx.badRequest('draggedId cannot equal targetId');
    }

    try {
      ctx.body = await service.reorderForAdmin(
        draggedId,
        Number.isFinite(targetId as number) ? (targetId as number) : null,
        position as 'child' | 'after' | 'root',
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to reorder category';
      if (message === 'Category not found') {
        return ctx.notFound(message);
      }
      throw error;
    }
  },

  async listTags(ctx: any) {
    const service = strapi.service('api::tag.tag') as any;
    ctx.body = await service.listForAdmin(ctx.query);
  },

  async findTag(ctx: any) {
    const service = strapi.service('api::tag.tag') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.findOneForAdmin(documentId, ctx.query);
    if (!data) {
      return ctx.notFound('Tag not found');
    }

    ctx.body = { data };
  },

  async createTag(ctx: any) {
    const service = strapi.service('api::tag.tag') as any;
    const data = await service.createForAdmin(ctx.request.body?.data);
    ctx.body = { data };
  },

  async updateTag(ctx: any) {
    const service = strapi.service('api::tag.tag') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.updateForAdmin(documentId, ctx.request.body?.data);
    ctx.body = { data };
  },

  async deleteTag(ctx: any) {
    const service = strapi.service('api::tag.tag') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    await service.deleteForAdmin(documentId);
    ctx.body = { data: { documentId } };
  },

  async mergeTags(ctx: any) {
    const service = strapi.service('api::tag.tag') as any;
    const sourceDocumentId = String(ctx.params?.sourceDocumentId ?? '').trim();
    const targetDocumentId = String(ctx.params?.targetDocumentId ?? '').trim();

    if (!sourceDocumentId || !targetDocumentId) {
      return ctx.badRequest('sourceDocumentId and targetDocumentId are required');
    }

    try {
      const data = await service.mergeForAdmin(sourceDocumentId, targetDocumentId);
      ctx.body = { data };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to merge tags';
      if (message === 'Tag not found') {
        return ctx.notFound(message);
      }
      if (message === 'Cannot merge a tag into itself') {
        return ctx.badRequest(message);
      }
      throw error;
    }
  },

  async listComments(ctx: any) {
    const service = strapi.service('api::comment.comment') as any;
    ctx.body = await service.listForAdmin(ctx.query);
  },

  async findComment(ctx: any) {
    const service = strapi.service('api::comment.comment') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.findOneForAdmin(documentId, ctx.query);
    if (!data) {
      return ctx.notFound('Comment not found');
    }

    ctx.body = { data };
  },

  async createComment(ctx: any) {
    const service = strapi.service('api::comment.comment') as any;
    const data = await service.createForAdmin(ctx.request.body?.data);
    ctx.body = { data };
  },

  async updateComment(ctx: any) {
    const service = strapi.service('api::comment.comment') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.updateForAdmin(documentId, ctx.request.body?.data);
    ctx.body = { data };
  },

  async deleteComment(ctx: any) {
    const service = strapi.service('api::comment.comment') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    await service.deleteForAdmin(documentId);
    ctx.body = { data: { documentId } };
  },

  async publishComment(ctx: any) {
    const service = strapi.service('api::comment.comment') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.publishForAdmin(documentId);
    ctx.body = { data };
  },

  async unpublishComment(ctx: any) {
    const service = strapi.service('api::comment.comment') as any;
    const documentId = readDocumentId(ctx);
    if (!documentId) return;

    const data = await service.unpublishForAdmin(documentId);
    ctx.body = { data };
  },

  async listUsers(ctx: any) {
    const page = Math.max(1, Number(ctx.query?.page ?? 1) || 1);
    const pageSize = Math.min(100, Math.max(1, Number(ctx.query?.pageSize ?? 10) || 10));
    const q = String(ctx.query?.q ?? '').trim();
    const isSeeded = String(ctx.query?.isSeeded ?? 'false').toLowerCase() === 'true';

    const service = strapi.service('api::admin-user.admin-user') as any;
    ctx.body = await service.list(page, pageSize, q, isSeeded);
  },

  async findUser(ctx: any) {
    const id = readUserId(ctx);
    if (!id) return;

    const service = strapi.service('api::admin-user.admin-user') as any;
    const user = await service.findOne(id);

    if (!user) {
      return ctx.notFound('User not found');
    }

    ctx.body = { data: user };
  },

  async listRoles(ctx: any) {
    const service = strapi.service('api::admin-user.admin-user') as any;
    const roles = await service.listRoles();
    ctx.body = { data: roles };
  },

  async createUser(ctx: any) {
    const service = strapi.service('api::admin-user.admin-user') as any;

    try {
      const user = await service.create(ctx.request.body?.data ?? {});
      ctx.body = { data: user };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create user';
      if (message === 'username, email and password are required') {
        return ctx.badRequest(message);
      }
      throw error;
    }
  },

  async updateUser(ctx: any) {
    const id = readUserId(ctx);
    if (!id) return;

    const service = strapi.service('api::admin-user.admin-user') as any;
    const user = await service.update(id, ctx.request.body?.data ?? {});
    ctx.body = { data: user };
  },

  async deleteUser(ctx: any) {
    const id = readUserId(ctx);
    if (!id) return;

    const service = strapi.service('api::admin-user.admin-user') as any;
    const data = await service.remove(id);
    ctx.body = { data };
  },

  async seedUsers(ctx: any) {
    const rawCount = ctx.request.body?.count;
    const resolved = rawCount === undefined || rawCount === null || rawCount === '' ? 20 : Number(rawCount);
    const count = Math.trunc(resolved);

    if (!Number.isFinite(count) || count < 1 || count > 500) {
      return ctx.badRequest('count must be an integer between 1 and 500');
    }

    const service = strapi.service('api::admin-user.admin-user') as any;
    const data = await service.seedAuthenticatedUsers(count);
    ctx.body = { data };
  },

  async batchDeleteSeedUsers(ctx: any) {
    const rawIds = Array.isArray(ctx.request.body?.ids) ? ctx.request.body.ids : [];
    const ids = rawIds.map((value: unknown) => Number(value)).filter((value: number) => Number.isFinite(value));

    const service = strapi.service('api::admin-user.admin-user') as any;
    const data = await service.batchDeleteSeedUsers(ids);
    ctx.body = { data };
  },

  async dashboard(ctx: any) {
    const service = strapi.service('api::admin-dashboard.admin-dashboard') as any;
    const data = await service.overview();
    ctx.body = { data };
  },

  async getAiAutomationSettings(ctx: any) {
    const data = await getAiAutomationSettings(strapi);
    ctx.body = { data };
  },

  async updateAiAutomationSettings(ctx: any) {
    try {
      const data = await updateAiAutomationSettings(strapi, ctx.request.body?.data ?? {});
      ctx.body = { data };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update AI automation settings';
      return ctx.badRequest(message);
    }
  },

  async checkAiProviderConnection(ctx: any) {
    try {
      const provider = String(ctx.request.body?.provider ?? '').trim();
      const apiKey = String(ctx.request.body?.apiKey ?? '').trim();
      const model = ctx.request.body?.model ? String(ctx.request.body.model).trim() : undefined;
      if (!provider || !apiKey) {
        return ctx.badRequest('provider and apiKey are required');
      }
      if (!['openai', 'anthropic'].includes(provider)) {
        return ctx.badRequest('provider must be openai or anthropic');
      }
      const data = await checkAiProviderConnection({
        provider: provider as 'openai' | 'anthropic',
        apiKey,
        model,
      });
      ctx.body = { data };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to check AI provider';
      return ctx.badRequest(message);
    }
  },

  async runAiContentCron(ctx: any) {
    try {
      const data = await runContentAutomation(strapi);
      ctx.body = { data };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to run AI content cron';
      return ctx.badRequest(message);
    }
  },

  async runAiCommentCron(ctx: any) {
    try {
      const data = await runCommentAutomation(strapi);
      ctx.body = { data };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to run AI comment cron';
      return ctx.badRequest(message);
    }
  },

  async testAiContent(ctx: any) {
    try {
      const data = await testContentAutomation(strapi);
      ctx.body = { data };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to test AI content';
      return ctx.badRequest(message);
    }
  },

  async testAiComment(ctx: any) {
    try {
      const data = await testCommentAutomation(strapi);
      ctx.body = { data };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to test AI comment';
      return ctx.badRequest(message);
    }
  },

  async triggerAutoEngage(ctx: any) {
    const { autoEngage } = await import('../../../cron/auto-engage');
    await autoEngage(strapi);
    ctx.body = { message: 'autoEngage triggered successfully' };
  },

  async listReports(ctx: any) {
    const page = Math.max(1, Number(ctx.query?.page ?? 1) || 1);
    const pageSize = Math.min(10, Math.max(1, Number(ctx.query?.pageSize ?? 10) || 10));
    const status = String(ctx.query?.status ?? '').trim();
    const targetType = String(ctx.query?.targetType ?? '').trim();

    const where: Record<string, unknown> = {};
    if (status && status !== 'all') where.status = status;
    if (targetType && targetType !== 'all') where.targetType = targetType;

    const [rawReports, total] = await Promise.all([
      strapi.db.query('api::report.report').findMany({
        where,
        orderBy: { createdAt: 'desc' },
        limit: pageSize,
        offset: (page - 1) * pageSize,
      }),
      strapi.db.query('api::report.report').count({ where }),
    ]);

    // Fetch reporters separately (populate on plugin::users-permissions.user is unreliable)
    const reporterIdSet = new Set<number>();
    for (const r of rawReports as any[]) {
      const rid = typeof r.reporter === 'object' ? r.reporter?.id : r.reporter;
      if (rid) reporterIdSet.add(rid);
    }
    const reporterIds = Array.from(reporterIdSet);
    const userRows = reporterIds.length > 0
      ? await strapi.db.query('plugin::users-permissions.user').findMany({
          where: { id: { $in: reporterIds } },
          select: ['id', 'username', 'email'],
        })
      : [];
    const userMap = new Map((userRows as any[]).map((u: any) => [u.id, u]));

    // Fetch post title+slug for post-type targets
    const postDocIds = [...new Set(
      (rawReports as any[]).filter((r: any) => r.targetType === 'post').map((r: any) => r.targetDocumentId).filter(Boolean),
    )] as string[];
    const postRows = postDocIds.length > 0
      ? await strapi.db.query('api::post.post').findMany({
          where: { documentId: { $in: postDocIds }, publishedAt: { $not: null } },
          select: ['documentId', 'title', 'slug'],
        })
      : [];
    const postMap = new Map((postRows as any[]).map((p: any) => [p.documentId, p]));

    const reports = (rawReports as any[]).map((r) => {
      const reporterId = typeof r.reporter === 'object' ? r.reporter?.id : r.reporter;
      const post = r.targetType === 'post' ? postMap.get(r.targetDocumentId) : null;
      return {
        ...r,
        reporter: reporterId ? (userMap.get(reporterId) ?? null) : null,
        targetTitle: post?.title ?? null,
        targetSlug: post?.slug ?? null,
      };
    });

    ctx.body = {
      data: reports,
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

  async updateReportStatus(ctx: any) {
    const id = Number(ctx.params?.id);
    if (!Number.isFinite(id)) return ctx.badRequest('Invalid id');

    const status = String(ctx.request.body?.status ?? '').trim();
    if (!['pending', 'reviewed', 'dismissed'].includes(status)) {
      return ctx.badRequest('status must be pending, reviewed, or dismissed');
    }

    const updated = await strapi.db.query('api::report.report').update({
      where: { id },
      data: { status },
    });

    if (!updated) return ctx.notFound('Report not found');
    ctx.body = { data: updated };
  },

  async deleteReport(ctx: any) {
    const id = Number(ctx.params?.id);
    if (!Number.isFinite(id)) return ctx.badRequest('Invalid id');

    await strapi.db.query('api::report.report').delete({ where: { id } });
    ctx.body = { data: { id } };
  },
};
