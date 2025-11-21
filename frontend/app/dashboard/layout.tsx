// FILE: frontend/app/dashboard/layout.tsx
import { AppSidebar } from '@/components/layout/AppSidebar';
import MobileSidebar from '@/components/layout/MobileSidebar';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen bg-[#050505] text-white overflow-hidden font-sans">
      <AppSidebar />
      <MobileSidebar /> {/* Added this */}

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full relative min-w-0 overflow-hidden bg-[#050505] pt-14 md:pt-0">
        {children}
      </main>
    </div>
  );
}