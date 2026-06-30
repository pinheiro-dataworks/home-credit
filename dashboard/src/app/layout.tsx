import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Home Credit Risk Dashboard",
  description: "Open-finance credit default risk prediction — powered by LightGBM + SHAP",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
