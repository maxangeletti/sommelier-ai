//
//  WineDetailView.swift
//  SommelierAI
//
//  Scheda dettaglio vino con design premium
//  - Hero section con score e match
//  - Animazioni smooth
//  - Layout responsive
//

import SwiftUI

struct WineDetailView: View {
    let wine: WineCard
    let userQuery: String
    
    @EnvironmentObject private var favoritesStore: FavoritesStore
    @EnvironmentObject private var tastingSheetStore: TastingSheetStore   // ✅ NEW
    @Environment(\.dismiss) private var dismiss
    
    @State private var isAppearing = false
    @State private var selectedTab: DetailTab = .overview
    @State private var showShareSheet = false
    @State private var showTastingSheet = false   // ✅ NEW
    
    private enum DetailTab: String, CaseIterable {
        case overview = "Panoramica"
        case tasting = "Degustazione"
        case pairing = "Abbinamenti"
        
        var icon: String {
            switch self {
            case .overview: return "info.circle.fill"
            case .tasting: return "wineglass.fill"
            case .pairing: return "fork.knife"
            }
        }
    }
    
    // Scores
    private var overallScore: Int {
        let score = max(0, min(5, wine.score ?? 0))
        return Int(score / 5.0 * 100)
    }
    
    private var matchScore: Int {
        let raw = (wine.match_score ?? wine.__match_score) ?? 0.0
        return Int(max(0, min(1, raw)) * 100)
    }
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // Hero Section
                heroSection
                    .opacity(isAppearing ? 1 : 0)
                    .offset(y: isAppearing ? 0 : 20)
                
                // Tabs
                tabSelector
                    .opacity(isAppearing ? 1 : 0)
                    .offset(y: isAppearing ? 0 : 10)
                
                // Content
                Group {
                    switch selectedTab {
                    case .overview:
                        overviewContent
                    case .tasting:
                        tastingContent
                    case .pairing:
                        pairingContent
                    }
                }
                .opacity(isAppearing ? 1 : 0)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                HStack(spacing: 16) {
                    // Degusta button
                    Button(action: {
                        showTastingSheet = true
                    }) {
                        Image(systemName: "list.clipboard")
                            .foregroundStyle(.secondary)
                            .font(.title3)
                    }
                    .accessibilityLabel("Scheda degustazione")
                    .accessibilityHint("Tocca per aprire la scheda di degustazione di questo vino")
                    
                    // Share button
                    Button(action: {
                        showShareSheet = true
                    }) {
                        Image(systemName: "square.and.arrow.up")
                            .foregroundStyle(.secondary)
                            .font(.title3)
                    }
                    .accessibilityLabel("Condividi vino")
                    .accessibilityHint("Tocca per condividere questo vino")
                    
                    // Favorite button
                    Button(action: {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                            favoritesStore.toggle(wine)
                        }
                    }) {
                        Image(systemName: favoritesStore.isFavorite(wine) ? "heart.fill" : "heart")
                            .foregroundStyle(favoritesStore.isFavorite(wine) ? .red : .secondary)
                            .font(.title3)
                    }
                    .accessibilityLabel(favoritesStore.isFavorite(wine) ? "Rimuovi dai preferiti" : "Aggiungi ai preferiti")
                    .accessibilityHint(favoritesStore.isFavorite(wine) ? "Tocca per rimuovere questo vino dai tuoi preferiti" : "Tocca per salvare questo vino nei tuoi preferiti")
                }
            }
        }
        .sheet(isPresented: $showShareSheet) {
            ShareSheet(items: WineShareHelper.shareItems(for: wine, query: userQuery))
        }
        .sheet(isPresented: $showTastingSheet) {
            NavigationStack {
                TastingSheetView(wine: wine)
            }
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.5)) {
                isAppearing = true
            }
        }
    }
    
    // MARK: - Hero Section
    
    private var heroSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Wine name
            Text(wine.name)
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(AppColors.textPrimary)
            
            // Producer + Region
            VStack(alignment: .leading, spacing: 4) {
                if let producer = wine.producer {
                    Text(producer)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(AppColors.textSecondary)
                }
                
                HStack(spacing: 4) {
                    if let country = wine.country {
                        Text(WineLocalizer.country(country))
                            .font(.subheadline)
                    }
                    if let region = wine.region {
                        Text("• \(WineLocalizer.region(region))")
                            .font(.subheadline)
                    }
                }
                .foregroundColor(AppColors.textSecondary)
            }
            
            // Scores row
            HStack(spacing: 20) {
                scoreCard(
                    title: "QUALITÀ",
                    score: overallScore,
                    color: AppColors.accentWine
                )
                
                scoreCard(
                    title: "MATCH",
                    score: matchScore,
                    color: .blue
                )
                
                if let price = wine.price {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("PREZZO")
                            .font(.caption2)
                            .foregroundColor(AppColors.textSecondary)
                        
                        Text(String(format: "€%.2f", price))
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(AppColors.primaryWine)
                    }
                }
            }
            
            // Badges
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    if wine.ottimo_valore == true {
                        pillBadge("💰 Ottimo Valore", color: .green)
                    }
                    
                    if (wine.rank ?? 99) == 1 {
                        pillBadge("🏆 Top Match", color: AppColors.primaryWine)
                    }
                    
                    if let vintage = wine.vintage {
                        pillBadge("\(vintage)", color: AppColors.textSecondary.opacity(0.8))
                    }
                    
                    if let rating = wine.rating_overall, rating > 4.0 {
                        pillBadge("⭐ \(String(format: "%.1f", rating))", color: .orange)
                    }
                }
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(AppColors.cardBackground)
                .shadow(color: .black.opacity(0.05), radius: 10, y: 5)
        )
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }
    
    // MARK: - Tab Selector
    
    private var tabSelector: some View {
        HStack(spacing: 0) {
            ForEach(DetailTab.allCases, id: \.self) { tab in
                Button(action: {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
                        selectedTab = tab
                    }
                }) {
                    VStack(spacing: 6) {
                        Image(systemName: tab.icon)
                            .font(.system(size: 18))
                        
                        Text(tab.rawValue)
                            .font(.system(size: 13, weight: .medium))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .foregroundColor(selectedTab == tab ? AppColors.primaryWine : AppColors.textSecondary)
                    .background(
                        selectedTab == tab ?
                        RoundedRectangle(cornerRadius: 12)
                            .fill(AppColors.primaryWine.opacity(0.1)) : nil
                    )
                }
                .accessibilityLabel("Tab \(tab.rawValue)")
                .accessibilityHint(selectedTab == tab ? "Selezionato" : "Tocca per visualizzare \(tab.rawValue.lowercased())")
            }
        }
        .padding(4)
        .background(AppColors.backgroundSecondary)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 16)
        .padding(.vertical, 16)
    }
    
    // MARK: - Overview Content
    
    private var overviewContent: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Perché questo vino
            section(title: "Perché questo vino") {
                if let explain = wine.explain?.first,
                   !explain.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text(cleanMarkdown(explain))
                        .font(.body)
                        .foregroundColor(AppColors.textPrimary)
                } else {
                    Text(cleanMarkdown(wine.reason))
                        .font(.body)
                        .foregroundColor(AppColors.textPrimary)
                }
            }
            
            // Vitigni
            if let grapes = wine.grapes, !grapes.isEmpty {
                section(title: "Vitigni") {
                    Text(grapes)
                        .font(.body)
                        .foregroundColor(AppColors.textPrimary)
                }
            }
            
            // Denominazione
            if let denom = wine.denomination {
                section(title: "Denominazione") {
                    Text(denom.uppercased())
                        .font(.headline)
                        .foregroundColor(AppColors.primaryWine)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 24)
    }
    
    // MARK: - Tasting Content
    
    private var tastingContent: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Aromi
            if let aromas = wine.aromas, !aromas.isEmpty {
                section(title: "Profilo aromatico") {
                    FlowLayout(spacing: 8) {
                        ForEach(aromas.prefix(10).map { $0 }, id: \.self) { aroma in
                            HStack(spacing: 6) {
                                Text(aromaEmoji(aroma))
                                    .font(.title3)
                                Text(aroma.replacingOccurrences(of: "_", with: " ").capitalized)
                                    .font(.subheadline)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(AppColors.cardBackground)
                            .clipShape(Capsule())
                        }
                    }
                }
            }
            
            // Caratteristiche
            section(title: "Caratteristiche") {
                VStack(alignment: .leading, spacing: 12) {
                    if let color = wine.color_detail {
                        charRow("Colore", WineLocalizer.color(color))
                    }
                    if let intensity = wine.intensity {
                        charRow("Intensità", WineLocalizer.intensity(intensity))
                    }
                    if let sweetness = wine.sweetness {
                        charRow("Dolcezza", WineLocalizer.sweetness(sweetness))
                    }
                    if let tannins = wine.tannins {
                        charRow("Tannini", WineLocalizer.tannin(tannins))
                    }
                    if let acidity = wine.acidity {
                        charRow("Acidità", WineLocalizer.acidity(acidity))
                    }
                }
            }
            
            // Giudizi
            if let quality = wine.quality {
                section(title: "Valutazione") {
                    VStack(alignment: .leading, spacing: 8) {
                        charRow("Qualità", quality.capitalized)
                        if let balance = wine.balance {
                            charRow("Equilibrio", balance.capitalized)
                        }
                        if let persistence = wine.persistence {
                            charRow("Persistenza", persistence.capitalized)
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 24)
    }
    
    // MARK: - Pairing Content
    
    private var pairingContent: some View {
        VStack(alignment: .leading, spacing: 20) {
            if let pairings = wine.food_pairings, !pairings.isEmpty {
                section(title: "Abbinamenti consigliati") {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(pairings.prefix(8).map { $0 }, id: \.self) { pairing in
                            HStack(spacing: 12) {
                                Text(pairingEmoji(pairing))
                                    .font(.system(size: 32))
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(pairing.replacingOccurrences(of: "_", with: " ").capitalized)
                                        .font(.body.weight(.medium))
                                        .foregroundColor(AppColors.textPrimary)
                                    
                                    Text("Abbinamento classico")
                                        .font(.caption)
                                        .foregroundColor(AppColors.textSecondary)
                                }
                                
                                Spacer()
                            }
                            .padding(12)
                            .background(AppColors.cardBackground)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                    }
                }
            } else {
                section(title: "Abbinamenti") {
                    Text("Informazioni non disponibili")
                        .font(.body)
                        .foregroundColor(AppColors.textSecondary)
                        .italic()
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 24)
    }
    
    // MARK: - Helpers
    
    @ViewBuilder
    private func section<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(AppColors.textPrimary)
            
            content()
        }
    }
    
    private func scoreCard(title: String, score: Int, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(AppColors.textSecondary)
            
            Text("\(score)%")
                .font(.system(size: 24, weight: .bold))
                .foregroundColor(color)
        }
    }
    
    private func pillBadge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.subheadline.weight(.medium))
            .foregroundColor(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(color)
            .clipShape(Capsule())
    }
    
    private func charRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(AppColors.textSecondary)
            
            Spacer()
            
            Text(value)
                .font(.subheadline.weight(.medium))
                .foregroundColor(AppColors.textPrimary)
        }
        .padding(.vertical, 4)
    }
    
    private func cleanMarkdown(_ text: String) -> String {
        text
            .replacingOccurrences(of: "# ", with: "")
            .replacingOccurrences(of: "## ", with: "")
            .replacingOccurrences(of: "### ", with: "")
            .replacingOccurrences(of: "**", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func aromaEmoji(_ aroma: String) -> String {
        let a = aroma.lowercased()
        
        // Frutti
        if a.contains("cilieg") || a.contains("cherry") { return "🍒" }
        if a.contains("fragol") || a.contains("strawberry") { return "🍓" }
        if a.contains("lampone") || a.contains("raspberry") { return "🫐" }
        if a.contains("mirtillo") || a.contains("blueberry") { return "🫐" }
        if a.contains("prugna") || a.contains("plum") { return "🍑" }
        if a.contains("pesca") || a.contains("peach") { return "🍑" }
        if a.contains("albicocca") || a.contains("apricot") { return "🍑" }
        if a.contains("mela") || a.contains("apple") { return "🍎" }
        if a.contains("pera") || a.contains("pear") { return "🍐" }
        if a.contains("agrumi") || a.contains("citrus") || a.contains("limone") { return "🍋" }
        if a.contains("arancia") || a.contains("orange") { return "🍊" }
        if a.contains("pompelmo") || a.contains("grapefruit") { return "🍊" }
        if a.contains("ananas") || a.contains("pineapple") { return "🍍" }
        if a.contains("banana") { return "🍌" }
        if a.contains("melone") || a.contains("melon") { return "🍈" }
        if a.contains("frutt") || a.contains("fruit") { return "🍇" }
        
        // Fiori
        if a.contains("fior") || a.contains("flor") || a.contains("rosa") { return "🌸" }
        if a.contains("viola") || a.contains("violet") { return "🌸" }
        if a.contains("gelsomino") || a.contains("jasmine") { return "🌺" }
        
        // Spezie & Aromi
        if a.contains("pepe") || a.contains("pepper") { return "🌶️" }
        if a.contains("vaniglia") || a.contains("vanilla") { return "🌿" }
        if a.contains("cannella") || a.contains("cinnamon") { return "🌿" }
        if a.contains("chiodi") || a.contains("clove") { return "🌿" }
        if a.contains("liquirizia") || a.contains("licorice") { return "🌿" }
        if a.contains("menta") || a.contains("mint") { return "🌿" }
        if a.contains("erbe") || a.contains("herb") || a.contains("basilico") { return "🌿" }
        
        // Legno & Terra
        if a.contains("legno") || a.contains("wood") || a.contains("oak") { return "🪵" }
        if a.contains("terra") || a.contains("earth") || a.contains("mineral") { return "🪨" }
        if a.contains("tabacco") || a.contains("tobacco") { return "🍂" }
        if a.contains("cuoio") || a.contains("leather") { return "🧳" }
        
        // Dolci & Tostati
        if a.contains("cacao") || a.contains("chocolate") || a.contains("cioccolat") { return "🍫" }
        if a.contains("caffè") || a.contains("coffee") { return "☕" }
        if a.contains("caramello") || a.contains("caramel") { return "🍮" }
        if a.contains("miele") || a.contains("honey") { return "🍯" }
        if a.contains("tostato") || a.contains("toasted") { return "🍞" }
        
        return "🍷"
    }
    
    private func pairingEmoji(_ pairing: String) -> String {
        let p = pairing.lowercased()
        
        if p.contains("carne") || p.contains("steak") || p.contains("beef") { return "🥩" }
        if p.contains("selvaggina") || p.contains("game") { return "🦌" }
        if p.contains("formag") || p.contains("cheese") { return "🧀" }
        if p.contains("pesce") || p.contains("fish") || p.contains("seafood") { return "🐟" }
        if p.contains("crostace") || p.contains("shellfish") { return "🦞" }
        if p.contains("sushi") { return "🍣" }
        if p.contains("pasta") { return "🍝" }
        if p.contains("pizza") { return "🍕" }
        if p.contains("barbecue") || p.contains("grill") { return "🍖" }
        if p.contains("aperitiv") { return "🥂" }
        if p.contains("dessert") || p.contains("dolc") { return "🍰" }
        
        return "🍽"
    }
}

// MARK: - Imperdibili Section (Similar Wines)
// TODO: Backend deve fornire array di vini simili nel response
// Placeholder per ora - da attivare quando backend è pronto
/*
private struct ImperdibiliSection: View {
    let similarWines: [SimilarWine]  // Array da backend
    
    var body: some View {
        if !similarWines.isEmpty {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .font(.title3)
                        .foregroundColor(AppColors.accentWine)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Gli Imperdibili")
                            .font(.title3.weight(.bold))
                            .foregroundColor(AppColors.textPrimary)
                        
                        Text("Vini simili che potrebbero piacerti")
                            .font(.caption)
                            .foregroundColor(AppColors.textSecondary)
                    }
                }
                .padding(.horizontal, 16)
                
                // Lista vini simili
                VStack(spacing: 12) {
                    ForEach(similarWines) { wine in
                        HStack {
                            // Numero/ID
                            Text("\(wine.id)")
                                .font(.headline)
                                .foregroundColor(AppColors.textSecondary)
                                .frame(width: 40, alignment: .leading)
                            
                            // Nome
                            Text(wine.name)
                                .font(.body)
                                .foregroundColor(AppColors.textPrimary)
                            
                            Spacer()
                            
                            // Prezzo
                            Text("€\(wine.price, specifier: "%.0f")")
                                .font(.body.weight(.semibold))
                                .foregroundColor(AppColors.accentWine)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(AppColors.cardBackground)
                        .cornerRadius(8)
                    }
                }
                .padding(.horizontal, 16)
            }
            .padding(.vertical, 20)
        }
    }
}

struct SimilarWine: Identifiable {
    let id: Int
    let name: String
    let price: Double
}
*/
