"use client";

import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { useAuth } from "./auth-context";

type SiteHeaderProps = {
  mobileMenuHref?: string;
};

type SuggestionHit = {
  id: number;
  documentId: string;
  title?: string;
  name?: string;
  slug: string;
  excerpt?: string;
  description?: string;
};

type Suggestions = {
  posts: SuggestionHit[];
  tags: SuggestionHit[];
  categories: SuggestionHit[];
};

const SearchIcon = ({ size = 16 }: { size?: number }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);

function stripHtml(html: string): string {
  return html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}

function SearchBox({ onSearch }: { onSearch: (q: string) => void }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<Suggestions | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.trim().length < 2) {
      setSuggestions(null);
      setShowDropdown(false);
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`/api/search-proxy?q=${encodeURIComponent(q.trim())}`);
      if (res.ok) {
        const data: Suggestions = await res.json();
        setSuggestions(data);
        setShowDropdown(true);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchSuggestions(query), 300);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [query, fetchSuggestions]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleSubmit = () => {
    if (!query.trim()) return;
    setShowDropdown(false);
    onSearch(query);
  };

  const handleSelect = (href: string) => {
    setShowDropdown(false);
    router.push(href);
  };

  const hasSuggestions = suggestions && (
    suggestions.posts.length > 0 || suggestions.tags.length > 0 || suggestions.categories.length > 0
  );

  return (
    <div ref={containerRef} className="relative w-full">
      <div className="flex items-center rounded-full border border-gray-300 bg-gray-100 transition-all focus-within:border-blue-400 focus-within:bg-white focus-within:ring-2 focus-within:ring-blue-500 dark:border-gray-600 dark:bg-gray-800 dark:focus-within:bg-gray-900">
        <span className="pointer-events-none pl-3 text-gray-400">
          <SearchIcon size={16} />
        </span>
        <input
          type="text"
          placeholder="Tìm kiếm..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          onFocus={() => hasSuggestions && setShowDropdown(true)}
          className="flex-1 bg-transparent py-2 pl-2 pr-1 text-sm text-gray-700 placeholder-gray-500 focus:outline-none dark:text-gray-200 dark:placeholder-gray-400"
        />
        <button
          type="button"
          onClick={handleSubmit}
          className="mr-1 flex items-center gap-1 rounded-full bg-gray-500 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-gray-600"
          aria-label="Tìm kiếm"
        >
          {loading ? (
            <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
          ) : (
            <SearchIcon size={11} />
          )}
          Tìm
        </button>
      </div>

      {showDropdown && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 overflow-hidden rounded-xl border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-800">
          {!hasSuggestions ? (
            <div className="px-4 py-3 text-sm text-gray-400">Không tìm thấy kết quả</div>
          ) : (
            <>
              {suggestions.posts.length > 0 && (
                <div>
                  <div className="border-b border-gray-100 px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-gray-400 dark:border-gray-700">Bài viết</div>
                  {suggestions.posts.slice(0, 4).map((post) => (
                    <button
                      key={post.documentId}
                      type="button"
                      className="flex w-full items-start gap-2 px-4 py-2.5 text-left hover:bg-gray-50 dark:hover:bg-gray-700"
                      onMouseDown={() => handleSelect(`/p/${post.slug}--${post.documentId}`)}
                    >
                      <SearchIcon size={14} />
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-gray-800 dark:text-gray-200">{post.title}</p>
                        {post.excerpt && <p className="truncate text-xs text-gray-400">{stripHtml(post.excerpt)}</p>}
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {suggestions.tags.length > 0 && (
                <div className="border-t border-gray-100 dark:border-gray-700">
                  <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-gray-400">Tags</div>
                  <div className="flex flex-wrap gap-1.5 px-4 pb-3">
                    {suggestions.tags.slice(0, 5).map((tag) => (
                      <button
                        key={tag.documentId}
                        type="button"
                        onMouseDown={() => handleSelect(`/t/${tag.slug}`)}
                        className="rounded-full bg-gray-100 px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
                      >
                        #{tag.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {suggestions.categories.length > 0 && (
                <div className="border-t border-gray-100 dark:border-gray-700">
                  <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wide text-gray-400">Danh mục</div>
                  {suggestions.categories.slice(0, 3).map((cat) => (
                    <button
                      key={cat.documentId}
                      type="button"
                      className="flex w-full items-center gap-2 px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700"
                      onMouseDown={() => handleSelect(`/c/${cat.slug}`)}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2z"/></svg>
                      {cat.name}
                    </button>
                  ))}
                </div>
              )}
              <div className="border-t border-gray-100 dark:border-gray-700">
                <button
                  type="button"
                  onMouseDown={handleSubmit}
                  className="flex w-full items-center justify-center gap-1.5 px-4 py-2.5 text-sm font-medium text-blue-600 hover:bg-blue-50 dark:text-blue-400 dark:hover:bg-blue-900/20"
                >
                  <SearchIcon size={13} />
                  Xem tất cả kết quả cho &quot;{query}&quot;
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function SiteHeader({ mobileMenuHref = "#mobile-left-drawer" }: SiteHeaderProps) {
  const { isLoggedIn, user, logout, openLoginModal } = useAuth();
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const isHydrated = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );

  useEffect(() => {
    const onDocClick = (event: MouseEvent) => {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const handleSearch = (q: string) => {
    router.push(`/search?q=${encodeURIComponent(q.trim())}`);
    setSearchOpen(false);
  };

  const showUserMenu = isHydrated && isLoggedIn && !!user;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-gray-200 bg-white shadow-md dark:border-gray-700 dark:bg-gray-900">
      <div className="container mx-auto flex h-16 items-center justify-between gap-4 px-4">
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            className="flex items-center gap-2 md:hidden"
            aria-label="Mở menu"
            onClick={() => document.dispatchEvent(new CustomEvent("open-right-drawer"))}
          >
            <div className="flex h-10 w-10 select-none items-center justify-center rounded-full bg-blue-600 text-lg font-bold text-white">
              T
            </div>
          </button>

          <Link href="/" className="hidden shrink-0 items-center gap-2 md:flex">
            <div className="flex h-10 w-10 select-none items-center justify-center rounded-full bg-blue-600 text-lg font-bold text-white">
              T
            </div>
            <span className="text-xl font-bold tracking-tight text-gray-900 dark:text-white">Trekky</span>
          </Link>
        </div>

        <div className="mx-4 hidden max-w-md flex-1 md:flex">
          <SearchBox onSearch={handleSearch} />
        </div>

        <button
          onClick={() => setSearchOpen(!searchOpen)}
          className="flex h-10 w-10 items-center justify-center rounded-full text-gray-600 hover:bg-gray-100 md:hidden dark:text-gray-300 dark:hover:bg-gray-800"
          aria-label="Mở tìm kiếm"
        >
          <SearchIcon size={20} />
        </button>

        <div className="flex shrink-0 items-center gap-2">
          {!isHydrated ? (
            <div className="h-10 w-[140px]" />
          ) : showUserMenu && user ? (
            <>
              <Link
                href="/my-posts/new"
                className="hidden items-center gap-1.5 rounded-full bg-gray-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-gray-600 sm:flex"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14" />
                  <path d="M12 5v14" />
                </svg>
                Tạo bài
              </Link>

              <Link
                href="/my-posts/new"
                className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-500 text-white sm:hidden"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14" />
                  <path d="M12 5v14" />
                </svg>
              </Link>

              <div ref={menuRef} className="relative">
                <button
                  type="button"
                  className="flex items-center gap-2 rounded-full border-2 border-gray-200 bg-gray-50 px-2 py-1 transition-colors hover:bg-gray-100 dark:border-gray-700 dark:bg-gray-800 dark:hover:bg-gray-700"
                  onClick={() => setMenuOpen((current) => !current)}
                >
                  {user.avatarUrl ? (
                    <Image
                      key={user.avatarUrl}
                      src={user.avatarUrl}
                      alt="Avatar"
                      width={32}
                      height={32}
                      className="h-8 w-8 rounded-full border-2 border-white object-cover"
                      unoptimized
                    />
                  ) : (
                    <div
                      className="flex h-8 w-8 items-center justify-center rounded-full border-2 border-white bg-gray-400 text-sm font-bold uppercase text-white"
                      title={user.email}
                    >
                      {(user.username ?? user.email)[0]}
                    </div>
                  )}
                  <span className="hidden text-sm font-medium text-gray-700 sm:block dark:text-gray-200">{user.username}</span>
                </button>

                {menuOpen && (
                  <div className="absolute right-0 mt-2 w-56 overflow-hidden rounded-xl border border-gray-200 bg-white shadow-xl dark:border-gray-700 dark:bg-gray-800">
                    <div className="border-b border-gray-100 px-4 py-3 dark:border-gray-700">
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{user.username}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">{user.email}</p>
                    </div>
                    <Link href="/my-posts/new" className="flex items-center gap-3 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700" onClick={() => setMenuOpen(false)}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="M12 5v14" /></svg>
                      Tạo bài viết
                    </Link>
                    <Link href="/my-posts" className="flex items-center gap-3 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700" onClick={() => setMenuOpen(false)}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" /></svg>
                      Bài viết của tôi
                    </Link>
                    <Link href="/profile/edit" className="flex items-center gap-3 px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700" onClick={() => setMenuOpen(false)}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>
                      Sửa hồ sơ
                    </Link>
                    <div className="border-t border-gray-100 dark:border-gray-700">
                      <button
                        type="button"
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20"
                        onClick={() => { setMenuOpen(false); logout(); router.push("/"); }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" x2="9" y1="12" y2="12" /></svg>
                        Đăng xuất
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <button
              onClick={openLoginModal}
              className="rounded-full bg-gray-500 px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-gray-600"
            >
              Đăng nhập
            </button>
          )}
        </div>
      </div>

      {searchOpen && (
        <div className="animate-fade-in absolute left-0 right-0 top-16 border-t border-gray-100 bg-white p-4 shadow-lg md:hidden dark:border-gray-700 dark:bg-gray-900">
          <SearchBox onSearch={handleSearch} />
        </div>
      )}
    </header>
  );
}
