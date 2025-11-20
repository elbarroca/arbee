export default function SettingsPage() {
    return (
      <div className="p-8 max-w-2xl">
        <h1 className="text-2xl font-bold text-white mb-8">Settings</h1>
        <div className="space-y-6">
          <div className="p-6 rounded-xl border border-zinc-800 bg-zinc-900/30">
            <h3 className="text-lg font-medium text-zinc-200 mb-2">API Configuration</h3>
            <p className="text-sm text-zinc-500 mb-4">Manage your connection to Supabase and Convo AI.</p>
            <div className="flex items-center justify-between p-3 bg-black rounded border border-zinc-800">
              <span className="text-sm text-zinc-400">Convo API Status</span>
              <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded border border-emerald-500/30">Active</span>
            </div>
          </div>
        </div>
      </div>
    );
  }