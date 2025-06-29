import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Quantum Platform - Next-Generation Quantum Computing',
  description: 'A comprehensive quantum computing platform with circuit design, simulation, and hardware execution capabilities.',
  keywords: ['quantum computing', 'quantum circuits', 'quantum simulation', 'quantum programming'],
  authors: [{ name: 'Quantum Platform Team' }],
  creator: 'Quantum Platform',
  publisher: 'Quantum Platform',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://quantum-platform.vercel.app',
    title: 'Quantum Platform - Next-Generation Quantum Computing',
    description: 'A comprehensive quantum computing platform with circuit design, simulation, and hardware execution capabilities.',
    siteName: 'Quantum Platform',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Quantum Platform - Next-Generation Quantum Computing',
    description: 'A comprehensive quantum computing platform with circuit design, simulation, and hardware execution capabilities.',
    creator: '@quantumplatform',
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
  },
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#000000' },
  ],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
      </head>
      <body className={`${inter.className} antialiased`}>
        <Providers>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                style: {
                  background: '#059669',
                },
              },
              error: {
                duration: 5000,
                style: {
                  background: '#dc2626',
                },
              },
            }}
          />
        </Providers>
      </body>
    </html>
  )
} 