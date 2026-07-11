import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEURIM",
  description: "Prompt-first EEG-guided image search with live FAA reward.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
