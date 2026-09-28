import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

export default function StatCard({ label, value, colorClass = 'text-white', trend = 'none' }) {
  
  const renderTrend = () => {
    if (trend === 'up') return <ArrowUpRight className="w-4 h-4 text-brand-accent" />;
    if (trend === 'down') return <ArrowDownRight className="w-4 h-4 text-green-500" />;
    if (trend === 'flat') return <Minus className="w-4 h-4 text-gray-500" />;
    return null;
  };

  return (
    <div className="bg-brand-surface border border-brand-border rounded-xl p-6 flex flex-col justify-center">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-400">{label}</span>
        {renderTrend()}
      </div>
      <div className={`text-3xl font-bold tracking-tight ${colorClass}`}>
        {value}
      </div>
    </div>
  );
}
