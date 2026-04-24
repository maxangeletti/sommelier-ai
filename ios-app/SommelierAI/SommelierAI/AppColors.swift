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
    static let primaryWine = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 139/255, green: 50/255, blue: 65/255, alpha: 1)  // Dark: Bordeaux luminoso
            : UIColor(red: 90/255, green: 29/255, blue: 46/255, alpha: 1)   // Light: Bordeaux scuro
    })
    
    // Rosso vino accento - saturo in dark
    static let accentWine = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 185/255, green: 60/255, blue: 75/255, alpha: 1)  // Dark: Rosso brillante
            : UIColor(red: 139/255, green: 30/255, blue: 45/255, alpha: 1)  // Light: Rosso vino
    })
    
    // --- GOLD / CHAMPAGNE ACCENTS ---
    // Oro principale - "brilla" su dark
    static let gold = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 230/255, green: 185/255, blue: 135/255, alpha: 1) // Dark: Oro luminoso
            : UIColor(red: 212/255, green: 165/255, blue: 116/255, alpha: 1) // Light: Oro medio
    })
    
    // Oro chiaro - per highlights
    static let goldLight = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 245/255, green: 210/255, blue: 165/255, alpha: 1) // Dark: Oro soft brillante
            : UIColor(red: 234/255, green: 198/255, blue: 154/255, alpha: 1) // Light: Oro pastello
    })
    
    // --- BACKGROUNDS ---
    // Background principale - crema → marrone scuro caldo
    static let backgroundPrimary = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 28/255, green: 20/255, blue: 16/255, alpha: 1)    // Dark: Marrone molto scuro
            : UIColor(red: 245/255, green: 235/255, blue: 217/255, alpha: 1) // Light: Crema splash
    })
    
    // Background secondario - per sezioni/cards elevation
    static let backgroundSecondary = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 40/255, green: 32/255, blue: 26/255, alpha: 1)    // Dark: Marrone medio scuro
            : UIColor(red: 236/255, green: 224/255, blue: 204/255, alpha: 1) // Light: Crema scuro
    })
    
    // Card background - bianco → quasi nero con warmth
    static let cardBackground = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 35/255, green: 28/255, blue: 23/255, alpha: 1)    // Dark: Quasi nero caldo
            : UIColor.white                                                  // Light: Bianco puro
    })
    
    // --- ACCENT COLORS ---
    // Blu accento - più saturo in dark
    static let blueAccent = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 70/255, green: 130/255, blue: 180/255, alpha: 1)  // Dark: Steel blue luminoso
            : UIColor(red: 47/255, green: 95/255, blue: 143/255, alpha: 1)   // Light: Blu medio
    })
    
    // Success green - più vivido in dark
    static let successGreen = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 100/255, green: 160/255, blue: 120/255, alpha: 1) // Dark: Verde brillante
            : UIColor(red: 74/255, green: 124/255, blue: 89/255, alpha: 1)   // Light: Verde scuro
    })
    
    // --- TEXT COLORS ---
    // Testo primario - marrone scuro → beige chiaro
    static let textPrimary = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 242/255, green: 235/255, blue: 225/255, alpha: 1) // Dark: Crema chiaro
            : UIColor(red: 44/255, green: 24/255, blue: 16/255, alpha: 1)    // Light: Dark brown
    })
    
    // Testo secondario - medium brown → beige medio
    static let textSecondary = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 200/255, green: 185/255, blue: 170/255, alpha: 1) // Dark: Beige
            : UIColor(red: 107/255, green: 90/255, blue: 76/255, alpha: 1)   // Light: Medium brown
    })
    
    // Testo muted - light brown → grigio caldo
    static let textMuted = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 140/255, green: 130/255, blue: 120/255, alpha: 1) // Dark: Grigio caldo
            : UIColor(red: 155/255, green: 138/255, blue: 124/255, alpha: 1) // Light: Light brown
    })
    
    // --- BORDERS ---
    // Border light - quasi invisibile ma presente
    static let borderLight = Color(uiColor: UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 60/255, green: 52/255, blue: 46/255, alpha: 1)    // Dark: Marrone border
            : UIColor(red: 232/255, green: 222/255, blue: 208/255, alpha: 1) // Light: Crema border
    })
}
