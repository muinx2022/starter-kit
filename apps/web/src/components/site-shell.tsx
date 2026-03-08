import { ReactNode } from "react";
import { RightSidebar } from "@/components/right-sidebar";
import { SiteHeader } from "@/components/site-header";
import { LoginModal } from "@/components/login-modal";
import { LeftSidebar } from "@/components/left-sidebar";
import { MobileRightDrawer } from "@/components/mobile-right-drawer";
import { getFooterPages, getTopLevelCategories } from "@/lib/strapi";

export async function SiteShell({ children }: { children: ReactNode }) {
  let categories: Awaited<ReturnType<typeof getTopLevelCategories>> = [];
  let footerPages: Awaited<ReturnType<typeof getFooterPages>> = [];

  try {
    [categories, footerPages] = await Promise.all([
      getTopLevelCategories(),
      getFooterPages(),
    ]);
  } catch {
    categories = [];
    footerPages = [];
  }

  return (
    <>
      <SiteHeader />
      <LoginModal />

      <div id="mobile-left-drawer" className="mobile-left-drawer md:hidden" aria-modal="true" role="dialog">
        <a href="#" className="mobile-left-overlay" aria-label="Dong menu" />
        <aside className="mobile-left-panel">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-base font-semibold text-gray-800 dark:text-gray-100">Danh muc</h3>
            <a href="#" className="rounded-md p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700" aria-label="Dong">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6 6 18" />
                <path d="m6 6 12 12" />
              </svg>
            </a>
          </div>
          <LeftSidebar categories={categories} />
        </aside>
      </div>

      <MobileRightDrawer>
        <RightSidebar categories={categories} />
      </MobileRightDrawer>

      <div className="container mx-auto flex-1 px-4 py-6">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-12">
          <main className="space-y-4 md:col-span-8">{children}</main>

          <aside className="hidden md:col-span-4 md:block md:self-start md:sticky md:top-20 md:h-[calc(100vh-5rem)]">
            <RightSidebar categories={categories} footerPages={footerPages} />
          </aside>
        </div>
      </div>
    </>
  );
}
