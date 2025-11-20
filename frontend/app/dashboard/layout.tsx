import { AppSidebar } from '@/components/layout/AppSidebar';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen bg-[#050505] text-white overflow-hidden font-sans">
      <AppSidebar />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full relative min-w-0 overflow-hidden bg-[#050505]">
        {children}
      </main>
    </div>
  );
}