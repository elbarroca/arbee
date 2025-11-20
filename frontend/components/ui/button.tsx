import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-full text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 active:scale-95",
  {
    variants: {
      variant: {
        // High contrast, premium white button with a soft white glow
        default: 
          "bg-white text-black hover:bg-zinc-200 border border-transparent shadow-[0_0_30px_-10px_rgba(255,255,255,0.5)] hover:shadow-[0_0_30px_-5px_rgba(255,255,255,0.7)]",
        
        // Your brand's Electric Blue (good for 'Send', 'Analyze', etc)
        cyber: 
          "bg-blue-600 text-white hover:bg-blue-500 border border-blue-500/50 shadow-[0_0_20px_-5px_rgba(37,99,235,0.5)]",
        
        // Glassmorphic dark button for secondary actions
        outline:
          "border border-zinc-800 bg-black/40 hover:bg-zinc-900 hover:text-white text-zinc-300 backdrop-blur-sm",
        
        destructive:
          "bg-red-900/50 text-red-200 border border-red-900 hover:bg-red-900",
          
        ghost: 
          "hover:bg-zinc-800 hover:text-white text-zinc-400",
          
        link: 
          "text-blue-500 underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-6 py-2",
        sm: "h-8 px-4 text-xs",
        lg: "h-12 px-8 text-base",
        icon: "h-10 w-10 rounded-full",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }