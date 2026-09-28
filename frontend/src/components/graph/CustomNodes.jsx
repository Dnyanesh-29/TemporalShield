import { Handle, Position } from 'reactflow';

// Common wrapper for flagged animation
const NodeWrapper = ({ flagged, children }) => (
  <div className={`relative flex items-center justify-center transition-all ${flagged ? 'scale-110' : ''}`}>
    {flagged && (
      <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-60 pointer-events-none"></div>
    )}
    <div className={`relative z-10 flex items-center justify-center ${flagged ? 'shadow-[0_0_15px_rgba(239,68,68,0.8)]' : ''}`}>
      {children}
    </div>
  </div>
);

// Employee: blue hexagon
export const EmployeeNode = ({ data }) => (
  <NodeWrapper flagged={data.flagged}>
    <div className="w-12 h-12 bg-blue-600 flex items-center justify-center text-[10px] font-bold text-white clip-hexagon border-2 border-blue-400">
      {data.label}
    </div>
    <Handle type="source" position={Position.Bottom} className="opacity-0" />
    <Handle type="target" position={Position.Top} className="opacity-0" />
  </NodeWrapper>
);

// Account: white circle
export const AccountNode = ({ data }) => (
  <NodeWrapper flagged={data.flagged}>
    <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center text-[10px] font-bold text-black border-2 border-gray-300">
      {data.label}
    </div>
    <Handle type="source" position={Position.Bottom} className="opacity-0" />
    <Handle type="target" position={Position.Top} className="opacity-0" />
  </NodeWrapper>
);

// Customer: grey square
export const CustomerNode = ({ data }) => (
  <NodeWrapper flagged={data.flagged}>
    <div className="w-10 h-10 bg-gray-500 rounded flex items-center justify-center text-[10px] font-bold text-white border-2 border-gray-400">
      {data.label}
    </div>
    <Handle type="source" position={Position.Bottom} className="opacity-0" />
    <Handle type="target" position={Position.Top} className="opacity-0" />
  </NodeWrapper>
);

// Transaction: amber diamond
export const TransactionNode = ({ data }) => (
  <NodeWrapper flagged={data.flagged}>
    <div className="w-10 h-10 bg-amber-500 rotate-45 flex items-center justify-center border-2 border-amber-300">
      <div className="-rotate-45 text-[10px] font-bold text-white">
        {data.label}
      </div>
    </div>
    <Handle type="source" position={Position.Bottom} className="opacity-0" />
    <Handle type="target" position={Position.Top} className="opacity-0" />
  </NodeWrapper>
);

export const nodeTypes = {
  employee: EmployeeNode,
  account: AccountNode,
  customer: CustomerNode,
  transaction: TransactionNode,
};
