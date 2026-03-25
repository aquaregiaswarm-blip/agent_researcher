export default function StatCard({ label, value, className = '' }: { label: string; value: string; className?: string }) {
  return (
    <div className="p-3 bg-gray-50 rounded-lg text-center border border-gray-200">
      <div className="text-sm font-medium text-gray-500 mb-1">{label}</div>
      <div className={`text-lg font-semibold text-gray-900 ${className}`}>{value}</div>
    </div>
  );
}
