import { notFound } from "next/navigation";
import type { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import { getCommentsForTarget, getPostByDocumentId } from "@/lib/strapi";
import { RichTextWithLightbox } from "@/components/rich-text-with-lightbox";
import { PostViewGallery } from "@/components/post-view-gallery";
import { PostActions } from "@/components/post-actions";
import { GenericComments } from "@/components/generic-comments";
import { SITE_URL, SITE_NAME, stripHtml, truncate, toAbsoluteMediaUrl, extractFirstImageFromHtml, buildOgImages } from "@/lib/seo";

export const dynamic = "force-dynamic";

function relativeTime(dateStr: string) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(mins / 60);
  const days = Math.floor(hours / 24);
  const weeks = Math.floor(days / 7);
  const months = Math.floor(days / 30);
  const years = Math.floor(days / 365);
  if (mins < 1) return "vừa xong";
  if (mins < 60) return `${mins} phút trước`;
  if (hours < 24) return `${hours} giờ trước`;
  if (days < 7) return `${days} ngày trước`;
  if (weeks < 4) return `${weeks} tuần trước`;
  if (months < 12) return `${months} tháng trước`;
  return `${years} năm trước`;
}

type PostPageProps = {
  params: Promise<{ id: string }>;
};

export async function generateMetadata({ params }: PostPageProps): Promise<Metadata> {
  const { id } = await params;
  const documentId = id.includes("--") ? id.split("--").pop() : id;
  if (!documentId) return {};
  const post = await getPostByDocumentId(documentId);
  if (!post) return {};

  const description = truncate(stripHtml(post.content ?? ""), 160);
  const canonical = `${SITE_URL}/p/${post.slug}--${post.documentId}`;
  const imageUrl =
    toAbsoluteMediaUrl(post.images?.[0]?.url) ??
    extractFirstImageFromHtml(post.content ?? "");

  return {
    title: post.title,
    description,
    alternates: { canonical },
    openGraph: {
      type: "article",
      url: canonical,
      title: post.title,
      description,
      siteName: SITE_NAME,
      locale: "vi_VN",
      images: buildOgImages(imageUrl, post.title),
      publishedTime: post.publishedAt,
      authors: post.author?.username ? [`${SITE_URL}/u/${post.author.username}`] : undefined,
      tags: post.tags?.map((t) => t.name),
      section: post.categories?.[0]?.name,
    },
    twitter: {
      card: imageUrl ? "summary_large_image" : "summary",
      title: post.title,
      description,
      images: imageUrl ? [imageUrl] : undefined,
    },
  };
}

export default async function PostPage({ params }: PostPageProps) {
  const { id } = await params;

  const documentId = id.includes("--") ? id.split("--").pop() : id;

  if (!documentId) {
    notFound();
  }

  const post = await getPostByDocumentId(documentId);

  if (!post) {
    notFound();
  }

  const comments = await getCommentsForTarget("post", post.documentId);

  const formattedDate = post.createdAt
    ? relativeTime(post.createdAt)
    : post.publishedAt
      ? relativeTime(post.publishedAt)
      : null;
  const createdAtMs = post.createdAt ? new Date(post.createdAt).getTime() : null;
  const updatedAtMs = post.updatedAt ? new Date(post.updatedAt).getTime() : null;
  const isUpdated = !!createdAtMs && !!updatedAtMs && updatedAtMs - createdAtMs > 60 * 1000;
  const updatedLabel = isUpdated && post.updatedAt ? relativeTime(post.updatedAt) : null;
  const authorInitial = (post.author?.username ?? "?")[0]?.toUpperCase() ?? "?";
  const authorAvatarUrl = toAbsoluteMediaUrl(post.author?.avatar?.url);
  const categories = post.categories ?? [];
  const tags = post.tags ?? [];

  return (
    <article className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden dark:bg-gray-800 dark:border-gray-700">
      <div className="px-6 pt-5 pb-5">

        {/* 1. Categories */}
        {categories.length > 0 && (
          <div className="flex flex-wrap items-center gap-1 text-xs text-gray-500 mb-3">
            {categories.map((cat, i) => (
              <span key={cat.documentId} className="flex items-center gap-1">
                {i > 0 && <span className="text-gray-300 dark:text-gray-600">·</span>}
                <Link
                  href={`/c/${cat.slug}`}
                  className="hover:text-gray-700 hover:underline transition-colors font-medium dark:hover:text-gray-300"
                >
                  {cat.name}
                </Link>
              </span>
            ))}
          </div>
        )}

        {/* 2. Title */}
        <h1 className="text-2xl font-bold text-gray-900 leading-snug dark:text-gray-100">{post.title}</h1>

        {/* 3. Author + time */}
        <div className="flex items-center gap-2 mt-3">
          <Link href={post.author?.username ? `/u/${post.author.username}` : "#"} className="shrink-0">
            {authorAvatarUrl ? (
              <Image
                src={authorAvatarUrl}
                alt={post.author?.username ?? "User avatar"}
                width={32}
                height={32}
                className="w-8 h-8 rounded-full object-cover bg-gray-300 dark:bg-gray-600"
                unoptimized
              />
            ) : (
              <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-600 font-bold text-xs dark:bg-gray-600 dark:text-gray-300">
                {authorInitial}
              </div>
            )}
          </Link>
          <div className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400">
            <Link
              href={post.author?.username ? `/u/${post.author.username}` : "#"}
              className="font-medium text-gray-700 hover:text-gray-900 transition-colors dark:text-gray-300 dark:hover:text-gray-100"
            >
              {post.author?.username ?? "Ẩn danh"}
            </Link>
            {formattedDate && (
              <>
                <span className="text-gray-300 dark:text-gray-600">·</span>
                <span>{formattedDate}</span>
              </>
            )}
            {updatedLabel && (
              <>
                <span className="text-gray-300 dark:text-gray-600">•</span>
                <span>Cập nhật: {updatedLabel}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* 4. Gallery */}
      {(post.images?.length ?? 0) > 0 && (
        <div className="px-6 pb-5">
          <PostViewGallery images={post.images!} />
        </div>
      )}

      {/* 5. Content */}
      <div className="px-6 pb-6">
        <RichTextWithLightbox
          html={post.content}
          className="prose prose-sm max-w-none text-gray-700 dark:text-gray-300 [&_img]:rounded-lg [&_img]:my-3 [&_img]:max-w-full [&_a]:text-blue-600 [&_a:hover]:text-blue-800 dark:[&_a]:text-blue-400 dark:[&_a:hover]:text-blue-300 [&_pre]:bg-gray-50 dark:[&_pre]:bg-gray-900 [&_pre]:rounded-lg [&_pre]:p-4 [&_blockquote]:border-l-4 [&_blockquote]:border-gray-300 dark:[&_blockquote]:border-gray-600 [&_blockquote]:pl-4 [&_blockquote]:text-gray-600 dark:[&_blockquote]:text-gray-400"
        />
      </div>

      {/* 6. Tags */}
      {tags.length > 0 && (
        <div className="px-6 pb-5 flex flex-wrap gap-1.5">
          {tags.map((tag) => (
            <Link
              key={tag.documentId}
              href={`/t/${tag.slug}`}
              className="inline-flex items-center px-2.5 py-0.5 rounded-full bg-gray-100 text-xs text-gray-500 hover:bg-gray-200 hover:text-gray-700 transition-colors dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600 dark:hover:text-gray-200"
            >
              #{tag.name}
            </Link>
          ))}
        </div>
      )}

      {/* 7. Actions */}
      <div className="px-6 border-t border-gray-100 dark:border-gray-700">
        <PostActions targetType="post" targetDocumentId={post.documentId} />
      </div>

      {/* 8. Comments */}
      <div className="px-6 pb-6 border-t border-gray-100 dark:border-gray-700 pt-5">
        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">
          Bình luận{" "}
          {comments.length > 0 && (
            <span className="text-gray-400 font-normal">({comments.length})</span>
          )}
        </h2>
        <GenericComments
          targetType="post"
          targetDocumentId={post.documentId}
          initialComments={comments}
        />
      </div>
    </article>
  );
}
