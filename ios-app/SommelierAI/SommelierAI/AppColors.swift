//
//  AppColors.swift
//  SommelierAI
//
//  Created by Massimiliano Angeletti on 27/02/26.
//


import SwiftUI

enum AppColors {
    
    // ✅ Palette allineata a icona e splash (3 Aprile 2026)
    
    // Rossi vino (primari)
    static let primaryWine = Color(red: 90/255, green: 29/255, blue: 46/255)      // #5A1D2E - Bordeaux icona
    static let accentWine  = Color(red: 139/255, green: 30/255, blue: 45/255)   // #8B1E2D - Rosso vino
    
    // Oro/Champagne (accenti premium)
    static let gold = Color(red: 212/255, green: 165/255, blue: 116/255)        // #D4A574 - Oro icona
    static let goldLight = Color(red: 234/255, green: 198/255, blue: 154/255)   // #EAC69A - Oro chiaro
    
    // Background (tema chiaro come splash)
    static let backgroundPrimary = Color(red: 245/255, green: 235/255, blue: 217/255)   // #F5EBD9 - Crema splash
    static let backgroundSecondary = Color(red: 236/255, green: 224/255, blue: 204/255) // #ECE0CC - Crema più scuro
    
    // Card
    static let cardBackground = Color.white
    
    // Blu accento (dalla palette icona)
    static let blueAccent = Color(red: 47/255, green: 95/255, blue: 143/255)    // #2F5F8F - Blu icona
    
}