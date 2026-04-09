//
//  WineDetailView.swift
//  SommelierAI
//
//  Screen completo dettaglio vino con:
//  - Tasting Notes (LLM)
//  - Similar Wines "Gli Imperdibili"
//

import SwiftUI

struct WineDetailView: View {
    let wine: WineCard
    let userQuery: String
    
    @State private var tastingNotes: String = ""
    @State private var isLoadingNotes: Bool = false
    @State private var similarWines: [WineCard] = []
    @State private var isLoadingSimilar: Bool = false
    
    @EnvironmentObject private var favoritesStore: FavoritesStore
    private let api = APIClient()
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                
                // Header
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text(wine.name)
                            .font(.title2.weight(.bold))
                        
                        Spacer()
                        
                        Button {
                            favoritesStore.toggle(wine)
                        } label: {
                            Image(systemName: favoritesStore.isFavorite(wine) ? "heart.fill" : "heart")
                                .font(.title3)
                                .foregroundStyle(favoritesStore.isFavorite(wine) ? .red : .secondary)
                        }
                    }
                    
                    if let price = wine.price {
                        Text(String(format: "€%.2f", price))
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(AppColors.accentWine)
                    }
                    
                    // Meta info
                    if let producer = wine.producer {
                        Text(producer)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    
                    let region = [wine.country, wine.region, wine.zone].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
                    if !region.isEmpty {
                        Text(region)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    
                    if let denomination = wine.denomination {
                        Text(denomination.uppercased())
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(AppColors.cardBackground)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                
                // Badge Ottimo Valore
                if wine.ottimo_valore == true {
                    HStack(spacing: 6) {
                        Image(systemName: "star.fill")
                        Text("Ottimo Valore")
                            .font(.subheadline.weight(.semibold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(Color.green)
                    .clipShape(Capsule())
                }
                
                // Aromi
                if let aromas = wine.aromas, !aromas.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Profilo aromatico")
                            .font(.headline)
                        
                        FlowLayout(spacing: 12) {
                            ForEach(aromas, id: \.self) { aroma in
                                HStack(spacing: 6) {
                                    Image(systemName: aromaIcon(for: aroma))
                                        .font(.callout)
                                        .foregroundStyle(AppColors.primaryWine)
                                    Text(aroma.capitalized)
                                        .font(.subheadline)
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(Color.gray.opacity(0.1))
                                .clipShape(Capsule())
                            }
                        }
                    }
                    .padding()
                    .background(AppColors.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                // Tasting Notes
                VStack(alignment: .leading, spacing: 12) {
                    Text("Note di degustazione")
                        .font(.headline)
                    
                    if isLoadingNotes {
                        ProgressView()
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding()
                    } else if !tastingNotes.isEmpty {
                        Text(tastingNotes)
                            .font(.body)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Caricamento...")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(AppColors.cardBackground)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                
                // Food Pairings
                if let pairings = wine.food_pairings, !pairings.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Abbinamenti consigliati")
                            .font(.headline)
                        
                        FlowLayout(spacing: 8) {
                            ForEach(pairings, id: \.self) { pairing in
                                Text("🍽 " + pairing.replacingOccurrences(of: "_", with: " ").capitalized)
                                    .font(.subheadline)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(Color.orange.opacity(0.1))
                                    .clipShape(Capsule())
                            }
                        }
                    }
                    .padding()
                    .background(AppColors.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                // Similar Wines
                if !similarWines.isEmpty {
                    VStack(alignment: .leading, spacing: 16) {
                        HStack {
                            Image(systemName: "sparkles")
                                .foregroundStyle(AppColors.gold)
                            Text("Gli Imperdibili")
                                .font(.headline)
                        }
                        
                        Text("Vini simili che potrebbero piacerti")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        
                        ForEach(similarWines) { similar in
                            SimilarWineRow(wine: similar)
                        }
                    }
                    .padding()
                    .background(AppColors.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                } else if isLoadingSimilar {
                    VStack {
                        ProgressView()
                        Text("Caricamento vini simili...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                }
                
                // Characteristics
                if let quality = wine.quality {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Caratteristiche")
                            .font(.headline)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            if let q = wine.quality {
                                CharacteristicRow(label: "Qualità", value: q)
                            }
                            if let b = wine.balance {
                                CharacteristicRow(label: "Equilibrio", value: b)
                            }
                            if let p = wine.persistence {
                                CharacteristicRow(label: "Persistenza", value: p)
                            }
                            if let i = wine.intensity {
                                CharacteristicRow(label: "Intensità", value: i.capitalized)
                            }
                            if let c = wine.color_detail {
                                CharacteristicRow(label: "Colore", value: c)
                            }
                        }
                    }
                    .padding()
                    .background(AppColors.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                // Purchase Link
                if let url = wine.purchase_url, !url.isEmpty, let validURL = URL(string: url) {
                    Link(destination: validURL) {
                        HStack {
                            Image(systemName: "cart.fill")
                            Text("Acquista online")
                                .font(.subheadline.weight(.semibold))
                        }
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(AppColors.accentWine)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }
            }
            .padding()
        }
        .background(AppColors.backgroundPrimary)
        .navigationTitle("Dettaglio")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await loadTastingNotes()
            await loadSimilarWines()
        }
    }
    
    private func loadTastingNotes() async {
        isLoadingNotes = true
        defer { isLoadingNotes = false }
        
        do {
            let notes = try await api.getTastingNotes(wineId: wine.id, query: userQuery)
            tastingNotes = notes
        } catch {
            tastingNotes = "Non disponibili al momento."
        }
    }
    
    private func loadSimilarWines() async {
        isLoadingSimilar = true
        defer { isLoadingSimilar = false }
        
        do {
            let wines = try await api.getSimilarWines(wineId: wine.id, limit: 3)
            similarWines = wines
        } catch {
            similarWines = []
        }
    }
    
    private func aromaIcon(for aroma: String) -> String {
        switch aroma.lowercased() {
        case "agrumi": return "leaf.fill"
        case "frutta rossa": return "heart.fill"
        case "frutta nera": return "circle.fill"
        case "fiori": return "sparkles"
        case "spezie": return "flame.fill"
        case "vaniglia": return "moon.fill"
        case "tostato": return "cup.and.saucer.fill"
        case "erbaceo": return "leaf"
        case "minerale": return "mountain.2.fill"
        case "balsamico": return "wind"
        default: return "circle"
        }
    }
}

// MARK: - Supporting Views

struct SimilarWineRow: View {
    let wine: WineCard
    
    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(wine.name)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(2)
                
                if let producer = wine.producer {
                    Text(producer)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            
            Spacer()
            
            if let price = wine.price {
                Text(String(format: "€%.0f", price))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(AppColors.accentWine)
            }
        }
        .padding(12)
        .background(Color.gray.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

struct CharacteristicRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            Spacer()
            
            Text(value)
                .font(.subheadline.weight(.medium))
        }
    }
}

// MARK: - FlowLayout (per aromi e abbinamenti)

struct FlowLayout: Layout {
    var spacing: CGFloat = 8
    
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = FlowResult(in: proposal.replacingUnspecifiedDimensions().width, subviews: subviews, spacing: spacing)
        return CGSize(width: proposal.width ?? 0, height: result.height)
    }
    
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = FlowResult(in: bounds.width, subviews: subviews, spacing: spacing)
        for (index, subview) in subviews.enumerated() {
            let position = result.positions[index]
            subview.place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y), proposal: .unspecified)
        }
    }
    
    struct FlowResult {
        var positions: [CGPoint] = []
        var height: CGFloat = 0
        
        init(in maxWidth: CGFloat, subviews: Subviews, spacing: CGFloat) {
            var x: CGFloat = 0
            var y: CGFloat = 0
            var lineHeight: CGFloat = 0
            
            for subview in subviews {
                let size = subview.sizeThatFits(.unspecified)
                
                if x + size.width > maxWidth && x > 0 {
                    x = 0
                    y += lineHeight + spacing
                    lineHeight = 0
                }
                
                positions.append(CGPoint(x: x, y: y))
                lineHeight = max(lineHeight, size.height)
                x += size.width + spacing
            }
            
            height = y + lineHeight
        }
    }
}
