//
//  Spacing.swift
//  SommelierAI
//
//  Design System v1.1 - Spacing Scale (Golden Ratio)
//  Updated: 24 Apr 2026 - More generous luxury spacing
//

import SwiftUI

enum Spacing {
    
    // MARK: - Spacing Scale (Golden Ratio ~1.6)
    
    static let xs: CGFloat = 4      // Tight spacing
    static let sm: CGFloat = 8      // Small gaps
    static let md: CGFloat = 12     // Medium gaps
    static let lg: CGFloat = 20     // Standard padding (+4 for more respiro)
    static let xl: CGFloat = 32     // Section spacing (+8)
    static let xxl: CGFloat = 48    // Large sections (+16)
    static let xxxl: CGFloat = 64   // Hero sections (NEW)
    
    // MARK: - Common Use Cases
    
    static let cardPadding: CGFloat = 20        // Card padding (+4 for luxury feel)
    static let screenEdges: CGFloat = 24        // Screen horizontal (+4)
    static let sectionSpacing: CGFloat = 32     // Between sections (+8)
    static let heroSpacing: CGFloat = 48        // Hero sections (NEW)
    
}
