import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { Button, type ButtonSize, type ButtonVariant } from './Button'

export type IconButtonProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  'aria-label' | 'children'
> & {
  'aria-label': string
  children: ReactNode
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
}

export function IconButton({
  variant = 'ghost',
  size = 'md',
  loading = false,
  children,
  ...rest
}: IconButtonProps) {
  return (
    <Button
      variant={variant}
      size={size}
      loading={loading}
      iconOnly
      icon={children}
      {...rest}
    />
  )
}
