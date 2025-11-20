import * as React from "react"
import { cn } from "@/lib/utils"

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          // Layout & Typography
          "flex h-11 w-full rounded-xl px-4 py-2 text-sm transition-all duration-200",
          
          // Colors & Background (Glassy Dark Theme)
          "bg-zinc-900/40 backdrop-blur-sm text-zinc-100",
          "border border-white/10",
          "placeholder:text-zinc-600",
          
          // Hover State
          "hover:bg-zinc-900/60 hover:border-white/20",
          
          // Focus State (Glow Effect)
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/20 focus-visible:border-blue-500/50",
          
          // Disabled State
          "disabled:cursor-not-allowed disabled:opacity-50",
          
          // File Input Styles
          "file:border-0 file:bg-transparent file:text-sm file:font-medium",
          
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }