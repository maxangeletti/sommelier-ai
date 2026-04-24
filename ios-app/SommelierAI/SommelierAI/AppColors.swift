//
//  AppColors.swift
//  SommelierAI
//
//  Created by Massimiliano Angeletti on 27/02/26.
//  Updated: Dark Mode Wine Theme - 24 Apr 2026
//

import SwiftUI

enum AppColors {
    
    // =========================
    // 🌙 DARK MODE WINE THEME
    // =========================
    // Sistema di colori adattivi per light/dark mode
    // Design: Wine elegante con accenti oro/champagne
    
    // --- PRIMARY WINE COLORS ---
    // Bordeaux principale - più luminoso in dark mode
    static let primaryWine = Color("PrimaryWine", bundle: nil, fallback:
        Color(light: Color(red: 90/255, green: 29/255, blue: 46/255),      // #5A1D2E - Light: Bordeaux scuro
              dark:  Color(red: 139/255, green: 50/255, blue: 65/255))     // #8B3241 - Dark: Bordeaux più luminoso
    )
    
    // Rosso vino accento - saturo in dark
    static let accentWine = Color("AccentWine", bundle: nil, fallback:
        Color(light: Color(red: 139/255, green: 30/255, blue: 45/255),     // #8B1E2D - Light: Rosso vino
              dark:  Color(red: 185/255, green: 60/255, blue: 75/255))     // #B93C4B - Dark: Rosso più brillante
    )
    
    // --- GOLD / CHAMPAGNE ACCENTS ---
    // Oro principale - "brilla" su dark
    static let gold = Color("Gold", bundle: nil, fallback:
        Color(light: Color(red: 212/255, green: 165/255, blue: 116/255),   // #D4A574 - Light: Oro medio
              dark:  Color(red: 230/255, green: 185/255, blue: 135/255))   // #E6B987 - Dark: Oro luminoso
    )
    
    // Oro chiaro - per highlights
    static let goldLight = Color("GoldLight", bundle: nil, fallback:
        Color(light: Color(red: 234/255, green: 198/255, blue: 154/255),   // #EAC69A - Light: Oro pastello
              dark:  Color(red: 245/255, green: 210/255, blue: 165/255))   // #F5D2A5 - Dark: Oro soft brillante
    )
    
    // --- BACKGROUNDS ---
    // Background principale - crema → marrone scuro caldo
    static let backgroundPrimary = Color("BackgroundPrimary", bundle: nil, fallback:
        Color(light: Color(red: 245/255, green: 235/255, blue: 217/255),   // #F5EBD9 - Light: Crema splash
              dark:  Color(red: 28/255, green: 20/255, blue: 16/255))      // #1C1410 - Dark: Marrone molto scuro
    )
    
    // Background secondario - per sezioni/cards elevation
    static let backgroundSecondary = Color("BackgroundSecondary", bundle: nil, fallback:
        Color(light: Color(red: 236/255, green: 224/255, blue: 204/255),   // #ECE0CC - Light: Crema scuro
              dark:  Color(red: 40/255, green: 32/255, blue: 26/255))      // #28201A - Dark: Marrone medio scuro
    )
    
    // Card background - bianco → quasi nero con warmth
    static let cardBackground = Color("CardBackground", bundle: nil, fallback:
        Color(light: .white,                                                // Light: Bianco puro
              dark:  Color(red: 35/255, green: 28/255, blue: 23/255))      // #231C17 - Dark: Quasi nero caldo
    )
    
    // --- ACCENT COLORS ---
    // Blu accento - più saturo in dark
    static let blueAccent = Color("BlueAccent", bundle: nil, fallback:
        Color(light: Color(red: 47/255, green: 95/255, blue: 143/255),     // #2F5F8F - Light: Blu medio
              dark:  Color(red: 70/255, green: 130/255, blue: 180/255))    // #4682B4 - Dark: Steel blue luminoso
    )
    
    // Success green - più vivido in dark
    static let successGreen = Color("SuccessGreen", bundle: nil, fallback:
        Color(light: Color(red: 74/255, green: 124/255, blue: 89/255),     // #4A7C59 - Light: Verde scuro
              dark:  Color(red: 100/255, green: 160/255, blue: 120/255))   // #64A078 - Dark: Verde più brillante
    )
    
    // --- TEXT COLORS ---
    // Testo primario - marrone scuro → beige chiaro
    static let textPrimary = Color("TextPrimary", bundle: nil, fallback:
        Color(light: Color(red: 44/255, green: 24/255, blue: 16/255),      // #2C1810 - Light: Dark brown
              dark:  Color(red: 242/255, green: 235/255, blue: 225/255))   // #F2EBE1 - Dark: Crema chiaro
    )
    
    // Testo secondario - medium brown → beige medio
    static let textSecondary = Color("TextSecondary", bundle: nil, fallback:
        Color(light: Color(red: 107/255, green: 90/255, blue: 76/255),     // #6B5A4C - Light: Medium brown
              dark:  Color(red: 200/255, green: 185/255, blue: 170/255))   // #C8B9AA - Dark: Beige
    )
    
    // Testo muted - light brown → grigio caldo
    static let textMuted = Color("TextMuted", bundle: nil, fallback:
        Color(light: Color(red: 155/255, green: 138/255, blue: 124/255),   // #9B8A7C - Light: Light brown
              dark:  Color(red: 140/255, green: 130/255, blue: 120/255))   // #8C8278 - Dark: Grigio caldo
    )
    
    // --- BORDERS ---
    // Border light - quasi invisibile ma presente
    static let borderLight = Color("BorderLight", bundle: nil, fallback:
        Color(light: Color(red: 232/255, green: 222/255, blue: 208/255),   // #E8DED0 - Light: Crema border
              dark:  Color(red: 60/255, green: 52/255, blue: 46/255))      // #3C342E - Dark: Marrone border
    )
}

// =========================
// HELPER: Color con light/dark
// =========================
extension Color {
    init(light: Color, dark: Color) {
        self.init(UIColor { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? UIColor(dark) : UIColor(light)
        })
    }
}
