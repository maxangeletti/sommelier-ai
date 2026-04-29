//
//  TastingSheet.swift
//  SommelierAI
//
//  Model per scheda degustazione professionale
//

import Foundation

struct TastingSheet: Codable, Identifiable {
    let id: UUID
    let wineId: String
    let wineName: String
    let date: Date
    
    // 👁️ Esame Visivo
    var visualColor: String?          // rosso rubino, giallo paglierino, ecc.
    var visualIntensity: Int?         // 1-5
    var visualClarity: String?        // limpido, velato, ecc.
    
    // 👃 Esame Olfattivo
    var olfactoryAromas: [String]     // array di aromi selezionati
    var olfactoryIntensity: Int?      // 1-5
    var olfactoryComplexity: Int?     // 1-5
    var olfactoryNotes: String?       // note libere
    
    // 👅 Esame Gustativo
    var gustatorySweetness: Int?      // 1-5
    var gustatoryAcidity: Int?        // 1-5
    var gustatoryTannins: Int?        // 1-5 (solo rossi)
    var gustatoryBody: Int?           // 1-5
    var gustatoryBalance: Int?        // 1-5
    var gustatoryPersistence: Int?    // 1-5
    var gustatoryNotes: String?       // note libere
    
    // 📝 Giudizio Finale
    var finalRating: Int?             // 1-5 stelle
    var finalNotes: String?           // note conclusive
    
    init(
        id: UUID = UUID(),
        wineId: String,
        wineName: String,
        date: Date = Date(),
        visualColor: String? = nil,
        visualIntensity: Int? = nil,
        visualClarity: String? = nil,
        olfactoryAromas: [String] = [],
        olfactoryIntensity: Int? = nil,
        olfactoryComplexity: Int? = nil,
        olfactoryNotes: String? = nil,
        gustatorySweetness: Int? = nil,
        gustatoryAcidity: Int? = nil,
        gustatoryTannins: Int? = nil,
        gustatoryBody: Int? = nil,
        gustatoryBalance: Int? = nil,
        gustatoryPersistence: Int? = nil,
        gustatoryNotes: String? = nil,
        finalRating: Int? = nil,
        finalNotes: String? = nil
    ) {
        self.id = id
        self.wineId = wineId
        self.wineName = wineName
        self.date = date
        self.visualColor = visualColor
        self.visualIntensity = visualIntensity
        self.visualClarity = visualClarity
        self.olfactoryAromas = olfactoryAromas
        self.olfactoryIntensity = olfactoryIntensity
        self.olfactoryComplexity = olfactoryComplexity
        self.olfactoryNotes = olfactoryNotes
        self.gustatorySweetness = gustatorySweetness
        self.gustatoryAcidity = gustatoryAcidity
        self.gustatoryTannins = gustatoryTannins
        self.gustatoryBody = gustatoryBody
        self.gustatoryBalance = gustatoryBalance
        self.gustatoryPersistence = gustatoryPersistence
        self.gustatoryNotes = gustatoryNotes
        self.finalRating = finalRating
        self.finalNotes = finalNotes
    }
}

// MARK: - Helpers

extension TastingSheet {
    
    /// Calcola se la scheda è completa
    var isComplete: Bool {
        finalRating != nil && !finalNotes.isNilOrEmpty
    }
    
    /// Percentuale completamento (0-100)
    var completionPercentage: Int {
        var completed = 0
        var total = 0
        
        // Visual (3 campi)
        total += 3
        if visualColor != nil { completed += 1 }
        if visualIntensity != nil { completed += 1 }
        if visualClarity != nil { completed += 1 }
        
        // Olfactory (4 campi)
        total += 4
        if !olfactoryAromas.isEmpty { completed += 1 }
        if olfactoryIntensity != nil { completed += 1 }
        if olfactoryComplexity != nil { completed += 1 }
        if olfactoryNotes != nil { completed += 1 }
        
        // Gustatory (7 campi)
        total += 7
        if gustatorySweetness != nil { completed += 1 }
        if gustatoryAcidity != nil { completed += 1 }
        if gustatoryTannins != nil { completed += 1 }
        if gustatoryBody != nil { completed += 1 }
        if gustatoryBalance != nil { completed += 1 }
        if gustatoryPersistence != nil { completed += 1 }
        if gustatoryNotes != nil { completed += 1 }
        
        // Final (2 campi)
        total += 2
        if finalRating != nil { completed += 1 }
        if finalNotes != nil { completed += 1 }
        
        return total > 0 ? (completed * 100) / total : 0
    }
}

// MARK: - String Extension

private extension String {
    var isNilOrEmpty: Bool {
        self.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

private extension Optional where Wrapped == String {
    var isNilOrEmpty: Bool {
        self?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true
    }
}
