export default function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white shadow-sm">
      <h4 className="text-base font-semibold text-gray-900 mb-4">{title}</h4>
      {children}
    </div>
  );
}
