import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Construction } from "lucide-react";

export default function WalletsPage() {
  return (
    <div className="p-8 flex flex-col h-full">
      <h1 className="text-2xl font-bold text-white mb-6">Wallet Tracker</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="bg-zinc-900/50 border-zinc-800 text-zinc-100">
           <CardHeader className="font-semibold">High Frequency Wallets</CardHeader>
           <CardContent className="text-4xl font-mono font-bold text-blue-400">142</CardContent>
        </Card>
        <Card className="bg-zinc-900/50 border-zinc-800 text-zinc-100">
           <CardHeader className="font-semibold">Whale Alerts (24h)</CardHeader>
           <CardContent className="text-4xl font-mono font-bold text-emerald-400">12</CardContent>
        </Card>
      </div>
      
      <div className="flex-1 flex items-center justify-center mt-12 border-2 border-dashed border-zinc-800 rounded-xl">
        <div className="text-center text-zinc-500">
          <Construction className="w-10 h-10 mx-auto mb-4 opacity-50" />
          <p>Detailed wallet table implementation coming next.</p>
        </div>
      </div>
    </div>
  );
}