'use client'

import { ReactNode } from 'react'
import { SessionProvider } from 'next-auth/react'
import { ErrorProvider } from '@/contexts/ErrorContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { QuantumProvider } from '@/contexts/QuantumContext'

interface ProvidersProps {
  children: ReactNode
}

export function Providers({ children }: ProvidersProps) {
  return (
    <SessionProvider>
      <ThemeProvider>
        <ErrorProvider>
          <QuantumProvider>
            {children}
          </QuantumProvider>
        </ErrorProvider>
      </ThemeProvider>
    </SessionProvider>
  )
} 