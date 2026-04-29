//
//  WineShareHelper.swift
//  SommelierAI
//
//  Helper per condivisione vini via ShareSheet iOS
//

import SwiftUI

struct WineShareHelper {
    
    /// Genera testo formattato per condivisione vino
    static func shareText(for wine: WineCard, query: String = "") -> String {
        var text = "🍷 \(wine.name)\n"
        
        if let producer = wine.producer {
            text += "\n📍 \(producer)"
        }
        
        if let region = wine.region {
            text += " · \(WineLocalizer.region(region))"
        }
        
        if let price = wine.price {
            text += "\n\n💰 €\(String(format: "%.2f", price))"
        }
        
        if let rating = wine.rating_overall, rating > 0 {
            let stars = String(repeating: "⭐", count: Int(rating.rounded()))
            text += "\n\n\(stars) \(String(format: "%.1f", rating))/5"
        }
        
        // Perché questo vino
        if let explain = wine.explain?.first, !explain.isEmpty {
            let clean = explain
                .replacingOccurrences(of: "# ", with: "")
                .replacingOccurrences(of: "## ", with: "")
                .replacingOccurrences(of: "### ", with: "")
                .replacingOccurrences(of: "**", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            text += "\n\n💡 \(clean)"
        } else {
            let cleanReason = wine.reason
                .replacingOccurrences(of: "# ", with: "")
                .replacingOccurrences(of: "## ", with: "")
                .replacingOccurrences(of: "### ", with: "")
                .replacingOccurrences(of: "**", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            text += "\n\n💡 \(cleanReason)"
        }
        
        if !query.isEmpty {
            text += "\n\n🔎 Trovato con: \"\(query)\""
        }
        
        text += "\n\n📱 Scoperto con Sommelier AI"
        
        return text
    }
    
    /// Crea ShareSheet items per vino
    static func shareItems(for wine: WineCard, query: String = "") -> [Any] {
        let text = shareText(for: wine, query: query)
        var items: [Any] = [text]
        
        // Se ha URL acquisto, aggiungilo
        if let url = wine.purchase_url, let validURL = URL(string: url) {
            items.append(validURL)
        }
        
        return items
    }
}

// MARK: - ShareSheet SwiftUI Wrapper

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: items,
            applicationActivities: nil
        )
        
        // Escludi activity non rilevanti
        controller.excludedActivityTypes = [
            .assignToContact,
            .addToReadingList,
            .openInIBooks
        ]
        
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
        // No updates needed
    }
}
