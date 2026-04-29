//
//  Typography.swift
//  SommelierAI
//
//  Design System v1.1 - Typography Scale with Dynamic Type
//  Updated: 27 Apr 2026 - Accessibility improvements
//

import SwiftUI

enum Typography {
    
    // MARK: - Headings (✅ Dynamic Type Enabled)
    
    // ✅ Usa text styles nativi che scalano automaticamente con le impostazioni utente
    static let largeTitle = Font.largeTitle.weight(.bold)    // Hero text (Welcome, splash) - Scala con user settings
    static let title1 = Font.title.weight(.bold)             // Screen titles - Scala con user settings
    static let title2 = Font.title2.weight(.bold)            // Section headers - Scala con user settings
    static let title3 = Font.title3.weight(.semibold)        // Card titles - Scala con user settings
    
    // MARK: - Body (✅ Dynamic Type Enabled)
    
    static let body = Font.body                              // Main text - Scala con user settings
    static let callout = Font.callout                        // Secondary text - Scala con user settings
    static let caption = Font.caption                        // Metadata, labels - Scala con user settings
    
    // MARK: - Special (⚠️ Custom sizes ma scalabili)
    
    // ✅ Per elementi custom (price, score), usa .system() con .leading per Dynamic Type
    static let price = Font.system(size: 24, weight: .bold, design: .default)
        .leading(.standard)  // ✅ Abilita line spacing automatico per accessibilità
    
    static let score = Font.system(size: 14, weight: .bold, design: .default)
        .leading(.standard)  // ✅ Abilita line spacing automatico
    
    // MARK: - Accessibility Notes
    // ✅ Tutti i font usano text styles nativi o .leading(.standard)
    // ✅ Testare con Settings → Display & Brightness → Text Size
    // ✅ Layout deve adattarsi a font grandi senza clipping
    
}
