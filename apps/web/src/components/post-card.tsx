import Image from "next/image";
import Link from "next/link";
import { RichTextContent } from "@/components/rich-text-content";
import { PostCardCarousel } from "@/components/post-card-carousel";
import { Post } from "@/lib/strapi";
import { toAbsoluteMediaUrl } from "@/lib/seo";

const VOID_TAGS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input",
  "link", "meta", "param", "source", "track", "wbr",
]);

function extractFirstMedia(content: string) {
  if (!content) return null;
  const videoMatch = content.match(/<(video|iframe)\b[^>]*>[\s\S]*?<\/\1>|<(video|iframe)\b[^>]*\/?\s*>/i);
  if (videoMatch?.[0]) return { type: "video" as const, html: videoMatch[0] };
  const imageMatch = content.match(/<img\b[^>]*\/?\s*>/i);
  if (imageMatch?.[0]) return { type: "image" as const, html: imageMatch[0] };
  return null;
}

function htmlExcerptByWords(html: string, limit = 80) {
  if (!html) return "";
  const tokens = html.match(/<[^>]+>|[^<]+/g) ?? [];
  const openTags: string[] = [];
  let words = 0;
  let reachedLimit = false;
  let result = "";

  for (const token of tokens) {
    if (reachedLimit) break;
    if (token.startsWith("<")) {
      const closing = token.match(/^<\s*\/\s*([a-zA-Z0-9-]+)/);
      if (closing) {
        const tagName = closing[1].toLowerCase();
        const stackIndex = openTags.lastIndexOf(tagName);
        if (stackIndex >= 0) { openTags.splice(stackIndex, 1); result += token; }
        continue;
      }
      const opening = token.match(/^<\s*([a-zA-Z0-9-]+)/);
      if (opening) {
        const tagName = opening[1].toLowerCase();
        result += token;
        if (!VOID_TAGS.has(tagName) && !/\/>\s*$/.test(token)) openTags.push(tagName);
      }
      continue;
    }
    const parts = token.match(/\S+|\s+/g) ?? [];
    let textChunk = "";
    for (const part of parts) {
      if (/^\s+$/.test(part)) { textChunk += part; continue; }
      if (words >= limit) { reachedLimit = true; break; }
      textChunk += part;
      words += 1;
    }
    result += textChunk;
    if (reachedLimit) result = result.trimEnd() + "...";
  }

  for (let i = openTags.length - 1; i >= 0; i -= 1) result += `</${openTags[i]}>`;
  return result;
}

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

export function PostCard({ post }: { post: Post }) {
  const formattedDate = post.createdAt
    ? relativeTime(post.createdAt)
    : post.publishedAt
      ? relativeTime(post.publishedAt)
      : null;
  const createdAtMs = post.createdAt ? new Date(post.createdAt).getTime() : null;
  const updatedAtMs = post.updatedAt ? new Date(post.updatedAt).getTime() : null;
  const isUpdated = !!createdAtMs && !!updatedAtMs && updatedAtMs - createdAtMs > 60 * 1000;
  const updatedLabel = isUpdated && post.updatedAt ? relativeTime(post.updatedAt) : null;
  const categories = post.categories ?? [];
  const tags = post.tags ?? [];
  const hasGallery = (post.images?.length ?? 0) > 0;
  const mediaPreview = !hasGallery ? extractFirstMedia(post.content) : null;
  const contentPreview = !hasGallery && !mediaPreview ? htmlExcerptByWords(post.content, 100) : "";
  const authorInitial = (post.author?.username ?? "?")[0].toUpperCase();
  const authorAvatarUrl = toAbsoluteMediaUrl(post.author?.avatar?.url);

  return (
    <article className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow dark:bg-gray-800 dark:border-gray-700">

      <div className="px-4 pt-4 pb-3">
        {/* 1. Categories */}
        {categories.length > 0 && (
          <div className="flex flex-wrap items-center gap-1 text-xs text-gray-400 mb-2">
            {categories.map((cat, i) => (
              <span key={cat.documentId} className="flex items-center gap-1">
                {i > 0 && <span className="text-gray-200 dark:text-gray-600">·</span>}
                <Link href={`/c/${cat.slug}`} className="hover:text-gray-600 hover:underline transition-colors font-medium dark:hover:text-gray-300">
                  {cat.name}
                </Link>
              </span>
            ))}
          </div>
        )}

        {/* 2. Title */}
        <Link
          href={`/p/${post.slug}--${post.documentId}`}
          className="block text-base font-bold text-gray-900 hover:text-gray-600 transition-colors line-clamp-2 leading-snug dark:text-gray-100 dark:hover:text-gray-300"
        >
          {post.title}
        </Link>

        {/* 3. Author + time */}
        <div className="flex items-center gap-1.5 mt-2">
          <Link href={post.author?.username ? `/u/${post.author.username}` : "#"} className="shrink-0">
            {authorAvatarUrl ? (
              <Image
                src={authorAvatarUrl}
                alt={post.author?.username ?? "User avatar"}
                width={24}
                height={24}
                className="w-6 h-6 rounded-full object-cover bg-gray-200"
                unoptimized
              />
            ) : (
              <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-gray-500 font-bold text-[10px] dark:bg-gray-600 dark:text-gray-300">
                {authorInitial}
              </div>
            )}
          </Link>
          <div className="flex items-center gap-1 text-xs text-gray-400">
            <Link href={post.author?.username ? `/u/${post.author.username}` : "#"} className="font-medium text-gray-600 hover:text-gray-800 transition-colors dark:text-gray-400 dark:hover:text-gray-200">
              {post.author?.username ?? "Ẩn danh"}
            </Link>
            {formattedDate && (
              <>
                <span className="text-gray-200 dark:text-gray-600">·</span>
                <span>{formattedDate}</span>
              </>
            )}
            {updatedLabel && (
              <>
                <span className="text-gray-200 dark:text-gray-600">•</span>
                <span>Cập nhật: {updatedLabel}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* 4. Media preview */}
      {hasGallery ? (
        <div className="mx-4 mb-3">
          <PostCardCarousel images={post.images!} />
        </div>
      ) : mediaPreview ? (
        <div className="mx-4 mb-3 rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-700">
          <RichTextContent
            html={mediaPreview.html}
            className="[&>iframe]:w-full [&>iframe]:aspect-video [&>iframe]:max-h-72 [&>img]:w-full [&>img]:object-cover [&>img]:max-h-72 [&>video]:w-full [&>video]:object-cover [&>video]:max-h-72"
          />
        </div>
      ) : contentPreview ? (
        <div className="px-4 mb-3 text-sm leading-relaxed text-gray-500 dark:text-gray-400">
          <RichTextContent html={contentPreview} />
        </div>
      ) : post.excerpt ? (
        <p className="px-4 mb-3 text-sm leading-relaxed text-gray-500 line-clamp-2 dark:text-gray-400">
          {post.excerpt}
        </p>
      ) : null}

      {/* 5. Tags */}
      {tags.length > 0 && (
        <div className="px-4 pb-3 flex flex-wrap gap-1">
          {tags.map((tag) => (
            <Link
              key={tag.documentId}
              href={`/t/${tag.slug}`}
              className="inline-flex items-center px-2 py-0.5 rounded-full bg-gray-100 text-xs text-gray-500 hover:bg-gray-200 hover:text-gray-700 transition-colors dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600 dark:hover:text-gray-200"
            >
              #{tag.name}
            </Link>
          ))}
        </div>
      )}

      {/* 6. Action bar */}
      <div className="px-4 py-2.5 border-t border-gray-100 flex items-center justify-between dark:border-gray-700">
        <div className="flex items-center gap-0.5">
          <button className="flex items-center gap-1 px-2.5 py-1.5 rounded-full hover:bg-pink-50 text-gray-400 hover:text-pink-500 transition-colors group dark:hover:bg-pink-900/20">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:scale-110 transition-transform">
              <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
            </svg>
            <span className="text-xs">{post.likesCount ?? 0}</span>
          </button>

          <button className="flex items-center gap-1 px-2.5 py-1.5 rounded-full hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors group dark:hover:bg-gray-700 dark:hover:text-gray-300">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:scale-110 transition-transform">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span className="text-xs">{post.commentsCount ?? 0}</span>
          </button>
        </div>

        <div className="flex items-center gap-0.5">
          <button className="flex items-center gap-1 px-2.5 py-1.5 rounded-full hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors dark:hover:bg-gray-700 dark:hover:text-gray-300">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
              <line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/>
            </svg>
            <span className="text-xs">Chia sẻ</span>
          </button>

          <Link
            href={`/p/${post.slug}--${post.documentId}`}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-full hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors dark:hover:bg-gray-700 dark:hover:text-gray-300"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>
            </svg>
            <span className="text-xs">Xem</span>
          </Link>
        </div>
      </div>
    </article>
  );
}
