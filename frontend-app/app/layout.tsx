import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEURIM Control",
  description: "Live NEURIM EEG reward, generated frame, and brain activity dashboard.",
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
