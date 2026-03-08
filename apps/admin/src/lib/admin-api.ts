"use client";

import { getStoredSession } from "@/lib/admin-auth";
import { nameGalleryFile } from "@/lib/media-naming";
import { uploadMediaFiles } from "@/lib/post-media";

const API_URL = "";
const DEFAULT_PAGE = 1;
const DEFAULT_PAGE_SIZE = 10;

export type PaginationMeta = {
  page: number;
  pageSize: number;
  pageCount: number;
  total: number;
};

export type PaginatedResult<T> = {
  data: T[];
  pagination: PaginationMeta;
};

export type PostItem = {
  id: number;
  documentId: string;
  title: string;
  slug: string;
  excerpt?: string;
  content?: string;
  createdAt?: string;
  updatedAt?: string;
  publishedAt?: string | null;
  categories?: Array<Pick<CategoryItem, "id" | "documentId" | "name" | "slug">>;
  tags?: Array<Pick<TagItem, "id" | "documentId" | "name" | "slug">>;
  author?: Pick<UserItem, "id" | "username" | "email"> | null;
  images?: MediaItem[];
  categoriesCount?: number;
  commentsCount?: number;
  aiSource?: {
    provider?: string;
    model?: string;
    generatedAt?: string;
  } | null;
};

export type PostInput = {
  title?: string;
  slug?: string;
  excerpt?: string;
  content?: string;
  categories?: string[];
  tags?: string[];
  author?: number | null;
  images?: number[];
};

export type PageType = "home" | "footer" | string;

export type PageItem = {
  id: number;
  documentId: string;
  title: string;
  slug: string;
  type: PageType;
  content?: string;
  createdAt?: string;
  updatedAt?: string;
};

export type PageInput = {
  title?: string;
  slug?: string;
  type?: PageType;
  content?: string;
};

export type MediaItem = {
  id: number;
  documentId?: string;
  url: string;
  mime?: string | null;
  alternativeText?: string | null;
  name?: string;
  width?: number | null;
  height?: number | null;
};

export type CategoryItem = {
  id: number;
  documentId: string;
  name: string;
  slug: string;
  description?: string;
  sortOrder?: number;
  createdAt?: string;
  updatedAt?: string;
  publishedAt?: string | null;
  parent?: { id: number; documentId: string; name: string } | null;
};

export type CategoryInput = {
  name?: string;
  slug?: string;
  description?: string;
  sortOrder?: number;
  parent?: string | null;
};

export type TagItem = {
  id: number;
  documentId: string;
  name: string;
  slug: string;
  description?: string;
  aliases?: string[];
  createdAt?: string;
  updatedAt?: string;
  publishedAt?: string | null;
};

export type TagInput = {
  name?: string;
  slug?: string;
  description?: string;
  aliases?: string[];
};

export type CommentTargetType = "post" | "page" | "product" | "hotel" | "tour" | "other";

export type CommentItem = {
  id: number;
  documentId: string;
  authorName: string;
  authorEmail?: string;
  content: string;
  targetType: CommentTargetType;
  targetDocumentId: string;
  targetTitle?: string;
  parent?: {
    id?: number;
    documentId: string;
    authorName?: string;
    targetType?: CommentTargetType;
    targetDocumentId?: string;
  } | null;
  createdAt?: string;
  updatedAt?: string;
  publishedAt?: string | null;
};

export type RoleItem = {
  id: number;
  name: string;
  type?: string;
};

export type UserItem = {
  id: number;
  username: string;
  email: string;
  blocked: boolean;
  confirmed: boolean;
  isSeeded?: boolean;
  role?: RoleItem | null;
};

export type AdminDashboardData = {
  totals: {
    posts: number;
    categories: number;
    comments: number;
  };
  recent: {
    posts: Array<Pick<PostItem, "id" | "documentId" | "title" | "slug" | "updatedAt">>;
    categories: Array<Pick<CategoryItem, "id" | "documentId" | "name" | "slug"> & { updatedAt?: string }>;
    comments: Array<
      Pick<CommentItem, "id" | "documentId" | "authorName" | "content" | "targetType" | "targetDocumentId"> & {
        updatedAt?: string;
      }
    >;
  };
};

export type AiAutomationJobStatus = {
  lastRunAt: string | null;
  lastSuccessAt: string | null;
  lastError: string | null;
};

export type AiContentCronSettings = AiAutomationJobStatus & {
  enabled: boolean;
  cron: string;
  postsPerRun: number;
  categoryDocumentIds: string[];
  contentPrompt: string;
  imageProvider: "auto" | "google" | "pexels";
  imageCountMin: number;
  imageCountMax: number;
  preferredMediaMode: "body" | "gallery";
};

export type AiCommentCronSettings = AiAutomationJobStatus & {
  enabled: boolean;
  cron: string;
  commentsPerRun: number;
  allowReplies: boolean;
  commentPrompt: string;
  replyPrompt: string;
};

export type AiProviderSettings = {
  enabled: boolean;
  apiKey: string;
  models: string[];
};

export type AiImageSearchProviderSettings = {
  enabled: boolean;
  apiKey: string;
  searchEngineId?: string;
};

export type AiAutomationSettings = {
  timezone: string;
  content: AiContentCronSettings;
  comments: AiCommentCronSettings;
  providers: {
    openai: AiProviderSettings;
    anthropic: AiProviderSettings;
  };
  imageSearch: {
    google: AiImageSearchProviderSettings;
    pexels: AiImageSearchProviderSettings;
  };
};

export type AiAutomationRunResult = {
  job: "content" | "comments";
  createdPosts?: number;
  uploadedImages?: number;
  embeddedBodyImages?: number;
  galleryImages?: number;
  createdComments?: number;
  skipped: number;
  errors: string[];
};

export type AiContentAutomationTestResult = {
  job: "content";
  provider: "openai" | "anthropic";
  model: string;
  category: {
    documentId: string;
    name: string;
    slug: string;
  };
  images: Array<{
    provider: "google" | "pexels";
    url: string;
    alt?: string;
  }>;
  preview: {
    title: string;
    excerpt: string;
    bodyText: string;
    imageSearchQueries: string[];
    relatedTags: string[];
    mediaMode: "body" | "gallery";
  };
  warnings: string[];
};

export type AiCommentAutomationTestResult = {
  job: "comments";
  provider: "openai" | "anthropic";
  model: string;
  post: {
    documentId: string;
    title: string;
  };
  replyMode: "top-level" | "reply";
  preview: string;
};

type ApiResponse<T> = { data: T } | T;

function toArray<T>(payload: ApiResponse<T[] | { data: T[] }>): T[] {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (payload && typeof payload === "object" && "data" in payload) {
    const data = (payload as { data: unknown }).data;
    if (Array.isArray(data)) {
      return data as T[];
    }
    if (data && typeof data === "object" && "data" in data) {
      const nested = (data as { data?: unknown }).data;
      if (Array.isArray(nested)) {
        return nested as T[];
      }
    }
  }
  return [];
}

function toItem<T>(payload: ApiResponse<T>): T {
  if (payload && typeof payload === "object" && "data" in payload) {
    return (payload as { data: T }).data;
  }
  return payload as T;
}

function toPagination(
  payload: unknown,
  page = DEFAULT_PAGE,
  pageSize = DEFAULT_PAGE_SIZE,
  total = 0,
): PaginationMeta {
  if (payload && typeof payload === "object" && "meta" in payload) {
    const meta = (payload as { meta?: unknown }).meta;
    if (meta && typeof meta === "object" && "pagination" in meta) {
      const pagination = (meta as { pagination?: Partial<PaginationMeta> }).pagination;
      if (pagination) {
        const resolvedTotal = Number.isFinite(pagination.total) ? Number(pagination.total) : total;
        const resolvedPageSize = Number.isFinite(pagination.pageSize)
          ? Number(pagination.pageSize)
          : pageSize;
        const resolvedPage = Number.isFinite(pagination.page) ? Number(pagination.page) : page;
        const resolvedPageCount = Number.isFinite(pagination.pageCount)
          ? Number(pagination.pageCount)
          : Math.max(1, Math.ceil((resolvedTotal || 0) / Math.max(1, resolvedPageSize)));

        return {
          page: resolvedPage,
          pageSize: resolvedPageSize,
          pageCount: resolvedPageCount,
          total: resolvedTotal,
        };
      }
    }
  }

  return {
    page,
    pageSize,
    pageCount: Math.max(1, Math.ceil(total / Math.max(1, pageSize))),
    total,
  };
}

function toPaginated<T>(
  payload: unknown,
  page = DEFAULT_PAGE,
  pageSize = DEFAULT_PAGE_SIZE,
): PaginatedResult<T> {
  const data = toArray<T>(payload as ApiResponse<T[] | { data: T[] }>);
  return {
    data,
    pagination: toPagination(payload, page, pageSize, data.length),
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const session = getStoredSession();
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(session?.jwt ? { Authorization: `Bearer ${session.jwt}` } : {}),
      ...(init?.headers ?? {}),
    },
  });

  const payload = (await response.json().catch(() => ({}))) as {
    error?: { message?: string };
  } & T;

  if (!response.ok) {
    throw new Error(payload.error?.message ?? `Request failed (${response.status})`);
  }

  return payload;
}

export async function listPosts(
  page = DEFAULT_PAGE,
  pageSize = DEFAULT_PAGE_SIZE,
  filters?: { q?: string; status?: "all" | "draft" | "published"; category?: string },
) {
  const query = new URLSearchParams({
    sort: "updatedAt:desc",
    "populate[categories][fields][0]": "id",
    "populate[categories][fields][1]": "documentId",
    "populate[categories][fields][2]": "name",
    "populate[categories][fields][3]": "slug",
    "populate[tags][fields][0]": "id",
    "populate[tags][fields][1]": "documentId",
    "populate[tags][fields][2]": "name",
    "populate[tags][fields][3]": "slug",
    "populate[author][fields][0]": "id",
    "populate[author][fields][1]": "username",
    "populate[author][fields][2]": "email",
    "populate[images][fields][0]": "id",
    "populate[images][fields][1]": "documentId",
    "populate[images][fields][2]": "url",
    "populate[images][fields][3]": "alternativeText",
    "populate[images][fields][4]": "width",
    "populate[images][fields][5]": "height",
    "populate[images][fields][6]": "name",
    "populate[images][fields][7]": "mime",
    "pagination[page]": String(page),
    "pagination[pageSize]": String(pageSize),
    ...(filters?.q?.trim() ? { q: filters.q.trim() } : {}),
    ...(filters?.status && filters.status !== "all" ? { status: filters.status } : {}),
    ...(filters?.category ? { "filters[categories][documentId][$eq]": filters.category } : {}),
  });
  const payload = await request(`/api/management/posts?${query.toString()}`);
  return toPaginated<PostItem>(payload, page, pageSize);
}

export async function getPost(documentId: string) {
  const query = new URLSearchParams({
    "populate[categories][fields][0]": "id",
    "populate[categories][fields][1]": "documentId",
    "populate[categories][fields][2]": "name",
    "populate[categories][fields][3]": "slug",
    "populate[tags][fields][0]": "id",
    "populate[tags][fields][1]": "documentId",
    "populate[tags][fields][2]": "name",
    "populate[tags][fields][3]": "slug",
    "populate[author][fields][0]": "id",
    "populate[author][fields][1]": "username",
    "populate[author][fields][2]": "email",
    "populate[images][fields][0]": "id",
    "populate[images][fields][1]": "documentId",
    "populate[images][fields][2]": "url",
    "populate[images][fields][3]": "alternativeText",
    "populate[images][fields][4]": "width",
    "populate[images][fields][5]": "height",
    "populate[images][fields][6]": "name",
    "populate[images][fields][7]": "mime",
  });
  const payload = await request<ApiResponse<PostItem>>(`/api/management/posts/${documentId}?${query.toString()}`);
  return toItem<PostItem>(payload);
}

export async function uploadImages(files: File[]) {
  if (files.length === 0) {
    return [] as MediaItem[];
  }

  return uploadMediaFiles<MediaItem>(files.map((file) => nameGalleryFile(file)));
}

export async function createPost(input: PostInput) {
  const payload = await request<ApiResponse<PostItem>>("/api/management/posts", {
    method: "POST",
    body: JSON.stringify({ data: input }),
  });
  return toItem<PostItem>(payload);
}

export async function updatePost(documentId: string, input: PostInput) {
  const payload = await request<ApiResponse<PostItem>>(`/api/management/posts/${documentId}`, {
    method: "PUT",
    body: JSON.stringify({ data: input }),
  });
  return toItem<PostItem>(payload);
}

export async function deletePost(documentId: string) {
  await request(`/api/management/posts/${documentId}`, { method: "DELETE" });
}

export async function publishPost(documentId: string) {
  const payload = await request<ApiResponse<PostItem>>(`/api/management/posts/${documentId}/publish`, {
    method: "POST",
  });
  return toItem<PostItem>(payload);
}

export async function unpublishPost(documentId: string) {
  const payload = await request<ApiResponse<PostItem>>(`/api/management/posts/${documentId}/unpublish`, {
    method: "POST",
  });
  return toItem<PostItem>(payload);
}

export async function listPages(page = DEFAULT_PAGE, pageSize = DEFAULT_PAGE_SIZE, q = "") {
  const query = new URLSearchParams({
    sort: "updatedAt:desc",
    "pagination[page]": String(page),
    "pagination[pageSize]": String(pageSize),
    ...(q.trim() ? { q: q.trim() } : {}),
  });
  const payload = await request(`/api/management/pages?${query.toString()}`);
  return toPaginated<PageItem>(payload, page, pageSize);
}

export async function getPage(documentId: string) {
  const payload = await request<ApiResponse<PageItem>>(`/api/management/pages/${documentId}`);
  return toItem<PageItem>(payload);
}

export async function createPage(input: PageInput) {
  const payload = await request<ApiResponse<PageItem>>("/api/management/pages", {
    method: "POST",
    body: JSON.stringify({ data: input }),
  });
  return toItem<PageItem>(payload);
}

export async function updatePage(documentId: string, input: PageInput) {
  const payload = await request<ApiResponse<PageItem>>(`/api/management/pages/${documentId}`, {
    method: "PUT",
    body: JSON.stringify({ data: input }),
  });
  return toItem<PageItem>(payload);
}

export async function deletePage(documentId: string) {
  await request(`/api/management/pages/${documentId}`, { method: "DELETE" });
}

export async function listCategories(page = DEFAULT_PAGE, pageSize = DEFAULT_PAGE_SIZE) {
  const query = new URLSearchParams({
    sort: "sortOrder:asc",
    "populate[parent][fields][0]": "id",
    "populate[parent][fields][1]": "documentId",
    "populate[parent][fields][2]": "name",
    "pagination[page]": String(page),
    "pagination[pageSize]": String(pageSize),
  });
  const payload = await request(`/api/management/categories?${query.toString()}`);
  return toPaginated<CategoryItem>(payload, page, pageSize);
}

export async function getCategory(documentId: string) {
  const query = new URLSearchParams({
    "populate[parent][fields][0]": "id",
    "populate[parent][fields][1]": "documentId",
    "populate[parent][fields][2]": "name",
  });
  const payload = await request<ApiResponse<CategoryItem>>(
    `/api/management/categories/${documentId}?${query.toString()}`,
  );
  return toItem<CategoryItem>(payload);
}

export async function listAllCategories() {
  const pageSize = 100;
  const first = await listCategories(1, pageSize);
  const all = [...first.data];

  for (let page = 2; page <= first.pagination.pageCount; page += 1) {
    const next = await listCategories(page, pageSize);
    all.push(...next.data);
  }

  return all;
}

export async function createCategory(input: CategoryInput) {
  const payload = await request<ApiResponse<CategoryItem>>("/api/management/categories", {
    method: "POST",
    body: JSON.stringify({ data: input }),
  });
  return toItem<CategoryItem>(payload);
}

export async function updateCategory(documentId: string, input: CategoryInput) {
  const payload = await request<ApiResponse<CategoryItem>>(`/api/management/categories/${documentId}`, {
    method: "PUT",
    body: JSON.stringify({ data: input }),
  });
  return toItem<CategoryItem>(payload);
}

export async function deleteCategory(documentId: string) {
  await request(`/api/management/categories/${documentId}`, { method: "DELETE" });
}

export async function publishCategory(documentId: string) {
  const payload = await request<ApiResponse<CategoryItem>>(`/api/management/categories/${documentId}/publish`, {
    method: "POST",
  });
  return toItem<CategoryItem>(payload);
}

export async function unpublishCategory(documentId: string) {
  const payload = await request<ApiResponse<CategoryItem>>(`/api/management/categories/${documentId}/unpublish`, {
    method: "POST",
  });
  return toItem<CategoryItem>(payload);
}

export async function reorderCategory(
  draggedId: number,
  targetId: number | null,
  position: "child" | "after" | "root",
) {
  const body: { draggedId: number; position: "child" | "after" | "root"; targetId?: number } = {
    draggedId,
    position,
  };
  if (typeof targetId === "number") {
    body.targetId = targetId;
  }

  await request("/api/management/categories/reorder", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function listTags(page = DEFAULT_PAGE, pageSize = DEFAULT_PAGE_SIZE, q = "") {
  const query = new URLSearchParams({
    sort: "updatedAt:desc",
    "pagination[page]": String(page),
    "pagination[pageSize]": String(pageSize),
    ...(q.trim() ? { q: q.trim() } : {}),
  });
  const payload = await request(`/api/management/tags?${query.toString()}`);
  return toPaginated<TagItem>(payload, page, pageSize);
}

export async function listAllTags() {
  const pageSize = 100;
  const first = await listTags(1, pageSize);
  const all = [...first.data];

  for (let page = 2; page <= first.pagination.pageCount; page += 1) {
    const next = await listTags(page, pageSize);
    all.push(...next.data);
  }

  return all;
}

export async function getTag(documentId: string) {
  const payload = await request<ApiResponse<TagItem>>(`/api/management/tags/${documentId}`);
  return toItem<TagItem>(payload);
}

export async function createTag(input: TagInput) {
  const payload = await request<ApiResponse<TagItem>>("/api/management/tags", {
    method: "POST",
    body: JSON.stringify({ data: input }),
  });
  return toItem<TagItem>(payload);
}

export async function updateTag(documentId: string, input: TagInput) {
  const payload = await request<ApiResponse<TagItem>>(`/api/management/tags/${documentId}`, {
    method: "PUT",
    body: JSON.stringify({ data: input }),
  });
  return toItem<TagItem>(payload);
}

export async function deleteTag(documentId: string) {
  await request(`/api/management/tags/${documentId}`, { method: "DELETE" });
}

export async function publishTag(documentId: string) {
  const payload = await request<ApiResponse<TagItem>>(`/api/management/tags/${documentId}/publish`, {
    method: "POST",
  });
  return toItem<TagItem>(payload);
}

export async function unpublishTag(documentId: string) {
  const payload = await request<ApiResponse<TagItem>>(`/api/management/tags/${documentId}/unpublish`, {
    method: "POST",
  });
  return toItem<TagItem>(payload);
}

export async function mergeTags(sourceDocumentId: string, targetDocumentId: string) {
  const payload = await request<ApiResponse<{ target: TagItem; mergedPostCount: number }>>(
    `/api/management/tags/${sourceDocumentId}/merge/${targetDocumentId}`,
    {
      method: "POST",
    },
  );
  return toItem<{ target: TagItem; mergedPostCount: number }>(payload);
}

export async function listComments(
  page = DEFAULT_PAGE,
  pageSize = DEFAULT_PAGE_SIZE,
  filters?: { q?: string; status?: "all" | "draft" | "published"; targetType?: string },
) {
  const query = new URLSearchParams({
    sort: "updatedAt:desc",
    "pagination[page]": String(page),
    "pagination[pageSize]": String(pageSize),
    ...(filters?.q?.trim() ? { q: filters.q.trim() } : {}),
    ...(filters?.status && filters.status !== "all" ? { status: filters.status } : {}),
    ...(filters?.targetType && filters.targetType !== "all"
      ? { "filters[targetType][$eq]": filters.targetType }
      : {}),
  });
  const payload = await request(`/api/management/comments?${query.toString()}`);
  return toPaginated<CommentItem>(payload, page, pageSize);
}

export async function listCommentsForTarget(
  targetType: CommentTargetType,
  targetDocumentId: string,
) {
  const pageSize = 200;
  const buildQuery = (page: number) =>
    new URLSearchParams({
      sort: "createdAt:asc",
      "filters[targetType][$eq]": targetType,
      "filters[targetDocumentId][$eq]": targetDocumentId,
      "pagination[page]": String(page),
      "pagination[pageSize]": String(pageSize),
    });

  const firstPayload = await request(`/api/management/comments?${buildQuery(1).toString()}`);
  const first = toPaginated<CommentItem>(firstPayload, 1, pageSize);
  const all = [...first.data];

  for (let page = 2; page <= first.pagination.pageCount; page += 1) {
    const nextPayload = await request(`/api/management/comments?${buildQuery(page).toString()}`);
    const next = toPaginated<CommentItem>(nextPayload, page, pageSize);
    all.push(...next.data);
  }

  return {
    data: all,
    pagination: {
      page: 1,
      pageSize: all.length || pageSize,
      pageCount: 1,
      total: all.length,
    },
  } satisfies PaginatedResult<CommentItem>;
}

export async function getComment(documentId: string) {
  const payload = await request<ApiResponse<CommentItem>>(`/api/management/comments/${documentId}`);
  return toItem<CommentItem>(payload);
}

export async function createComment(input: Partial<CommentItem>) {
  const payload = await request<ApiResponse<CommentItem>>("/api/management/comments", {
    method: "POST",
    body: JSON.stringify({ data: input }),
  });
  return toItem<CommentItem>(payload);
}

export async function updateComment(documentId: string, input: Partial<CommentItem>) {
  const payload = await request<ApiResponse<CommentItem>>(`/api/management/comments/${documentId}`, {
    method: "PUT",
    body: JSON.stringify({ data: input }),
  });
  return toItem<CommentItem>(payload);
}

export async function deleteComment(documentId: string) {
  await request(`/api/management/comments/${documentId}`, { method: "DELETE" });
}

export async function publishComment(documentId: string) {
  const payload = await request<ApiResponse<CommentItem>>(`/api/management/comments/${documentId}/publish`, {
    method: "POST",
  });
  return toItem<CommentItem>(payload);
}

export async function unpublishComment(documentId: string) {
  const payload = await request<ApiResponse<CommentItem>>(`/api/management/comments/${documentId}/unpublish`, {
    method: "POST",
  });
  return toItem<CommentItem>(payload);
}

export async function listUsers(
  page = DEFAULT_PAGE,
  pageSize = DEFAULT_PAGE_SIZE,
  q = "",
  isSeeded = false,
) {
  const query = new URLSearchParams({
    page: String(page),
    pageSize: String(pageSize),
    isSeeded: String(isSeeded),
    ...(q.trim() ? { q: q.trim() } : {}),
  });
  const payload = await request(`/api/management/users?${query.toString()}`);
  return toPaginated<UserItem>(payload, page, pageSize);
}

export async function getUser(id: number) {
  const payload = await request<ApiResponse<UserItem>>(`/api/management/users/${id}`);
  return toItem<UserItem>(payload);
}

export async function listRoles() {
  const payload = await request<ApiResponse<RoleItem[]>>("/api/management/roles");
  return toArray<RoleItem>(payload);
}

export async function createUser(input: {
  username: string;
  email: string;
  password: string;
  roleId?: number;
  blocked?: boolean;
  confirmed?: boolean;
}) {
  const payload = await request<ApiResponse<UserItem>>("/api/management/users", {
    method: "POST",
    body: JSON.stringify({ data: input }),
  });
  return toItem<UserItem>(payload);
}

export async function updateUser(
  id: number,
  input: {
    username?: string;
    email?: string;
    password?: string;
    roleId?: number;
    blocked?: boolean;
    confirmed?: boolean;
  },
) {
  const payload = await request<ApiResponse<UserItem>>(`/api/management/users/${id}`, {
    method: "PUT",
    body: JSON.stringify({ data: input }),
  });
  return toItem<UserItem>(payload);
}

export async function deleteUser(id: number) {
  await request(`/api/management/users/${id}`, { method: "DELETE" });
}

export async function seedUsers(count: number = 20) {
  const payload = await request<
    ApiResponse<{
      createdCount: number;
      startUsername: string | null;
      endUsername: string | null;
      defaultPasswordHint: string;
    }>
  >("/api/management/users/seed", {
    method: "POST",
    body: JSON.stringify({ count }),
  });
  return toItem(payload);
}

export async function getRandomSeedUser() {
  const first = await listUsers(1, 1, "", true);
  const total = first.pagination.total ?? 0;
  if (total <= 0) {
    return null;
  }

  const randomPage = Math.floor(Math.random() * total) + 1;
  const result = await listUsers(randomPage, 1, "", true);
  return result.data[0] ?? null;
}

export async function batchDeleteSeedUsers(ids: number[]) {
  const payload = await request<
    ApiResponse<{
      requestedCount: number;
      deletedCount: number;
      skippedCount: number;
    }>
  >("/api/management/users/seed/batch-delete", {
    method: "POST",
    body: JSON.stringify({ ids }),
  });
  return toItem(payload);
}

export async function getAdminDashboard() {
  const payload = await request<ApiResponse<AdminDashboardData>>("/api/management/dashboard");
  return toItem<AdminDashboardData>(payload);
}

export async function getAiAutomationSettings() {
  const payload = await request<ApiResponse<AiAutomationSettings>>("/api/management/settings/ai-automation");
  return toItem<AiAutomationSettings>(payload);
}

export async function updateAiAutomationSettings(input: Partial<AiAutomationSettings>) {
  const payload = await request<ApiResponse<AiAutomationSettings>>("/api/management/settings/ai-automation", {
    method: "PUT",
    body: JSON.stringify({ data: input }),
  });
  return toItem<AiAutomationSettings>(payload);
}

export async function runAiContentCron() {
  const payload = await request<ApiResponse<AiAutomationRunResult>>("/api/management/cron/content/run", {
    method: "POST",
  });
  return toItem<AiAutomationRunResult>(payload);
}

export async function runAiCommentCron() {
  const payload = await request<ApiResponse<AiAutomationRunResult>>("/api/management/cron/comments/run", {
    method: "POST",
  });
  return toItem<AiAutomationRunResult>(payload);
}

export async function checkAiProviderConnection(input: {
  provider: "openai" | "anthropic";
  apiKey: string;
  model?: string;
}) {
  const payload = await request<ApiResponse<{ ok: boolean; provider: string; model: string; message: string }>>(
    "/api/management/settings/ai-automation/check-provider",
    {
      method: "POST",
      body: JSON.stringify(input),
    },
  );
  return toItem(payload);
}

export async function testAiContent() {
  const payload = await request<ApiResponse<AiContentAutomationTestResult>>(
    "/api/management/settings/ai-automation/test-content",
    {
      method: "POST",
    },
  );
  return toItem(payload);
}

export async function testAiComment() {
  const payload = await request<ApiResponse<AiCommentAutomationTestResult>>(
    "/api/management/settings/ai-automation/test-comment",
    {
      method: "POST",
    },
  );
  return toItem(payload);
}

export async function triggerAutoEngage() {
  return request<{ message: string }>("/api/management/cron/auto-engage", { method: "POST" });
}

export type ReportStatus = "pending" | "reviewed" | "dismissed";

export type ReportItem = {
  id: number;
  targetType: string;
  targetDocumentId: string;
  targetTitle?: string | null;
  targetSlug?: string | null;
  reason?: string | null;
  status: ReportStatus;
  reporter?: { id: number; username: string; email: string } | null;
  createdAt?: string;
  updatedAt?: string;
};

export async function listReports(
  page = DEFAULT_PAGE,
  pageSize = DEFAULT_PAGE_SIZE,
  filters?: { status?: "all" | ReportStatus; targetType?: string },
) {
  const query = new URLSearchParams({
    page: String(page),
    pageSize: String(pageSize),
    ...(filters?.status && filters.status !== "all" ? { status: filters.status } : {}),
    ...(filters?.targetType && filters.targetType !== "all" ? { targetType: filters.targetType } : {}),
  });
  const payload = await request(`/api/management/reports?${query.toString()}`);
  return toPaginated<ReportItem>(payload, page, pageSize);
}

export async function updateReportStatus(id: number, status: ReportStatus) {
  const payload = await request<ApiResponse<ReportItem>>(`/api/management/reports/${id}/status`, {
    method: "PUT",
    body: JSON.stringify({ status }),
  });
  return toItem<ReportItem>(payload);
}

export async function deleteReport(id: number) {
  await request(`/api/management/reports/${id}`, { method: "DELETE" });
}
