import { Construction } from "lucide-react";

export default function ScannerPage() {
  return (
    <div className="p-8 h-full flex flex-col items-center justify-center text-zinc-500">
      <div className="w-16 h-16 bg-zinc-900 rounded-full flex items-center justify-center mb-6">
        <Construction className="w-8 h-8 text-blue-500" />
      </div>
      <h2 className="text-xl font-semibold text-white mb-2">Market Scanner</h2>
      <p>Real-time Polymarket event scanning module under construction.</p>
    </div>
  );
}