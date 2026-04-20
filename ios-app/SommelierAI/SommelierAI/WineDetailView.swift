//
//  WineDetailView.swift
//  SommelierAI
//
//  NUOVO DESIGN: Scheda dettaglio compatta e visual-first
//  - Score complessivo prominente
//  - Match % + Annata affiancati
//  - Abbinamenti con icone grandi
//  - Prezzo hero
//

import SwiftUI

struct WineDetailView: View {
    let wine: WineCard
    let userQuery: String
    
    @EnvironmentObject private var favoritesStore: FavoritesStore
    
    @State private var tastingNotes: String = ""
    @State private var isLoadingNotes: Bool = false
    @State private var similarWines: [WineCard] = []
    @State private var isLoadingSimilar: Bool = false
    
    private let api = APIClient()
    
    // Calcolo score complessivo (overall) - CLAMPED tra 0-100%
    private var overallScore: Int {
        let score = max(0, min(5, wine.score ?? 0))  // Clamp tra 0 e 5
        return Int(score / 5.0 * 100)
    }
    
    // Calcolo match score
    private var matchScore: Int {
        let raw = (wine.match_score ?? wine.__match_score) ?? 0.0
        return Int(max(0, min(1, raw)) * 100)
    }
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                
                // ✅ Badge contestuale in alto (es: "Rosso per bistecca")
                if let topBadge = contextualTopBadge() {
                    Text(topBadge)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(AppColors.accentWine)
                        .clipShape(Capsule())
                }
                
                // ✅ Badge "Eccellente scelta" (rosso)
                if wine.ottimo_valore == true || (wine.rank ?? 99) == 1 {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.subheadline)
                        Text("Eccellente scelta")
                            .font(.subheadline.weight(.semibold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.red)
                    .clipShape(Capsule())
                }
                
                // ✅ TITOLO VINO GRANDE
                Text(wine.name)
                    .font(.system(size: 32, weight: .bold))
                    .lineLimit(3)
                    .padding(.top, 8)
                
                // ✅ PUNTEGGIO COMPLESSIVO
                VStack(alignment: .leading, spacing: 4) {
                    Text("PUNTEGGIO COMPLESSIVO")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .tracking(1)
                    
                    Text("\(overallScore)%")
                        .font(.system(size: 48, weight: .bold))
                        .foregroundStyle(AppColors.accentWine)
                }
                .padding(.top, 4)
                
                // ✅ DUE COLONNE: MATCH + ANNATA
                HStack(spacing: 20) {
                    // Match
                    VStack(alignment: .leading, spacing: 4) {
                        Text("MATCH")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .tracking(1)
                        
                        Text("\(matchScore)%")
                            .font(.system(size: 28, weight: .bold))
                            .foregroundStyle(.primary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.gray.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    
                    // Annata
                    VStack(alignment: .leading, spacing: 4) {
                        Text("ANNATA")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .tracking(1)
                        
                        Text(wine.vintage.map { String($0) } ?? "N/A")
                            .font(.system(size: 28, weight: .bold))
                            .foregroundStyle(.primary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color.gray.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                
                // ✅ PRODUCER (con icona)
                if let producer = wine.producer, !producer.isEmpty {
                    HStack(spacing: 10) {
                        Image(systemName: "building.2.fill")
                            .font(.title3)
                            .foregroundStyle(AppColors.accentWine)
                        
                        Text(producer)
                            .font(.body)
                            .foregroundStyle(.primary)
                    }
                    .padding(.top, 8)
                }
                
                // ✅ LOCATION (con icona)
                let location = [wine.region, wine.country].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: ", ")
                if !location.isEmpty {
                    HStack(spacing: 10) {
                        Image(systemName: "location.fill")
                            .font(.title3)
                            .foregroundStyle(AppColors.accentWine)
                        
                        Text(location)
                            .font(.body)
                            .foregroundStyle(.primary)
                    }
                }
                
                Divider()
                    .padding(.vertical, 8)
                
                // ✅ ABBINAMENTI IDEALI (con emoji grandi)
                if let pairings = wine.food_pairings, !pairings.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("ABBINAMENTI IDEALI")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .tracking(1)
                        
                        HStack(spacing: 16) {
                            ForEach(Array(pairings.prefix(3)), id: \.self) { pairing in
                                VStack(spacing: 6) {
                                    Text(pairingEmoji(for: pairing))
                                        .font(.system(size: 40))
                                    
                                    Text(pairing.replacingOccurrences(of: "_", with: " ").capitalized)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .multilineTextAlignment(.center)
                                        .lineLimit(2)
                                }
                                .frame(maxWidth: .infinity)
                            }
                        }
                    }
                    .padding(.top, 8)
                }
                
                // ✅ CARATTERISTICHE PRINCIPALI (Tannicità + Acidità)
                if wine.tannins != nil || wine.acidity != nil {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("CARATTERISTICHE")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .tracking(1)
                        
                        HStack(spacing: 20) {
                            if let t = wine.tannins {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Tannicità")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                    Text(WineLocalizer.tannins(t))
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(.primary)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            
                            if let a = wine.acidity {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Acidità")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                    Text(WineLocalizer.acidity(a))
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(.primary)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                    .padding(.top, 8)
                }
                
                // 🍒 SEZIONE SENTORI (con emoji)
                if let aromas = wine.aromas, !aromas.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("SENTORI")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .tracking(1)
                        
                        HStack(spacing: 16) {
                            ForEach(Array(aromas.prefix(4)), id: \.self) { aroma in
                                VStack(spacing: 6) {
                                    Text(aromaEmoji(for: aroma))
                                        .font(.system(size: 40))
                                    Text(aroma.capitalized)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .multilineTextAlignment(.center)
                                        .lineLimit(2)
                                }
                                .frame(maxWidth: .infinity)
                            }
                        }
                    }
                    .padding(.top, 8)
                }
                
                Divider()
                    .padding(.vertical, 8)
                
                // ✅ STELLE (rating)
                if let rating = wine.rating_overall, rating > 0 {
                    HStack(spacing: 4) {
                        ForEach(0..<5, id: \.self) { i in
                            let clamped = max(0, min(5, rating))
                            let full = Int(clamped.rounded(.down))
                            let hasHalf = (clamped - Double(full)) >= 0.5
                            let colorFilled = Color(red: 0.95, green: 0.82, blue: 0.35)
                            
                            if i < full {
                                Image(systemName: "star.fill")
                                    .font(.title2)
                                    .foregroundStyle(colorFilled)
                            } else if i == full && hasHalf {
                                Image(systemName: "star.leadinghalf.filled")
                                    .font(.title2)
                                    .foregroundStyle(colorFilled)
                            } else {
                                Image(systemName: "star")
                                    .font(.title2)
                                    .foregroundStyle(Color.gray.opacity(0.3))
                            }
                        }
                    }
                }
                
                // ✅ PREZZO HERO
                if let price = wine.price {
                    Text(String(format: "€%.2f", price))
                        .font(.system(size: 32, weight: .bold))
                        .foregroundStyle(.primary)
                        .padding(.top, 12)
                }
                
                // ✅ BOTTONE "SCHEDA TECNICA" (rosso scuro)
                NavigationLink {
                    WineDetailExpandedView(wine: wine, userQuery: userQuery)
                } label: {
                    HStack {
                        Image(systemName: "doc.text.fill")
                            .font(.headline)
                        Text("Scheda tecnica")
                            .font(.headline)
                    }
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Color(red: 0.5, green: 0.1, blue: 0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                }
                .buttonStyle(.plain)
                .padding(.top, 8)
                
                // ✅ HEART BUTTON (floating)
                HStack {
                    Spacer()
                    Button {
                        favoritesStore.toggle(wine)
                    } label: {
                        Image(systemName: favoritesStore.isFavorite(wine) ? "heart.fill" : "heart")
                            .font(.title2)
                            .foregroundStyle(favoritesStore.isFavorite(wine) ? .red : .secondary)
                            .padding(12)
                            .background(Circle().fill(Color.white.opacity(0.9)))
                            .shadow(radius: 4)
                    }
                }
                .padding(.top, -600) // Floating position
            }
            .padding(20)
        }
        .background(Color(red: 0.98, green: 0.94, blue: 0.78).opacity(0.3))
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
    }
    
    // ✅ Badge contestuale basato su query + food pairings
    private func contextualTopBadge() -> String? {
        let q = userQuery.lowercased()
        let pairings = (wine.food_pairings ?? []).joined(separator: " ").lowercased()
        
        if q.contains("bistecca") || pairings.contains("steak") || pairings.contains("carne") {
            let colorIta = WineLocalizer.color(wine.color)
            if colorIta.lowercased() == "rosso" {
                return "Rosso per bistecca"
            }
        }
        
        if q.contains("pesce") || pairings.contains("pesce") || pairings.contains("seafood") {
            let colorIta = WineLocalizer.color(wine.color)
            if colorIta.lowercased() == "bianco" {
                return "Bianco per pesce"
            }
        }
        
        if q.contains("aperitivo") || pairings.contains("aperitivo") {
            return "Perfetto per aperitivo"
        }
        
        if q.contains("cena") {
            return "Ideale per cena importante"
        }
        
        return nil
    }
    
    // ✅ Emoji per food pairings
    private func pairingEmoji(for pairing: String) -> String {
        let p = pairing.lowercased()
        
        if p.contains("carne") || p.contains("steak") || p.contains("beef") {
            return "🥩"
        }
        if p.contains("selvaggina") || p.contains("game") {
            return "🦌"
        }
        if p.contains("formag") || p.contains("cheese") {
            return "🧀"
        }
        if p.contains("pesce") || p.contains("fish") || p.contains("seafood") {
            return "🐟"
        }
        if p.contains("crostace") || p.contains("shellfish") || p.contains("gamberi") || p.contains("aragosta") {
            return "🦞"
        }
        if p.contains("mollusc") || p.contains("oyster") || p.contains("ostriche") || p.contains("cozze") {
            return "🦪"
        }
        if p.contains("sushi") {
            return "🍣"
        }
        if p.contains("pasta") {
            return "🍝"
        }
        if p.contains("pizza") {
            return "🍕"
        }
        if p.contains("barbecue") || p.contains("grill") {
            return "🍖"
        }
        if p.contains("aperitiv") || p.contains("cocktail") {
            return "🥂"
        }
        if p.contains("dessert") || p.contains("dolc") {
            return "🍰"
        }
        if p.contains("vegetarian") || p.contains("verdur") {
            return "🥗"
        }
        
        return "🍽"
    }
    
    // 🍒 Emoji per aromi
    private func aromaEmoji(for aroma: String) -> String {
        let normalized = aroma.lowercased().trimmingCharacters(in: .whitespaces)
        
        // Frutta
        if normalized.contains("frutta nera") || normalized.contains("frutta_nera") { return "🍇" } // 💡 NUOVO: frutta nera generica
        if normalized.contains("uva") || normalized.contains("grape") { return "🍇" } // 💡 NUOVO: uva/uva nera
        if normalized.contains("ciliegi") || normalized.contains("cherry") { return "🍒" }
        if normalized.contains("fragol") || normalized.contains("strawberry") { return "🍓" }
        if normalized.contains("lampon") || normalized.contains("raspberry") { return "🥝" }
        if normalized.contains("mora") || normalized.contains("blackberry") { return "🫛" }
        if normalized.contains("mirtill") || normalized.contains("blueberry") { return "🫐" }
        if normalized.contains("prun") || normalized.contains("plum") { return "🍑" }
        if normalized.contains("ribes") { return "🍇" }
        if normalized.contains("mela") || normalized.contains("apple") { return "🍏" }
        if normalized.contains("pera") || normalized.contains("pear") { return "🍐" }
        if normalized.contains("pesca") || normalized.contains("peach") { return "🍑" }
        if normalized.contains("albicocc") || normalized.contains("apricot") { return "🥝" }
        if normalized.contains("agrumi") || normalized.contains("citrus") { return "🍋" }
        if normalized.contains("limon") || normalized.contains("lemon") { return "🍋" }
        
        // Fiori
        if normalized.contains("rosa") || normalized.contains("rose") { return "🌹" }
        if normalized.contains("viola") || normalized.contains("violet") { return "🌸" }
        if normalized.contains("gelsomino") || normalized.contains("jasmine") { return "🌼" }
        if normalized.contains("fiori") || normalized.contains("floral") { return "🌸" }
        
        // Spezie & Erbe
        if normalized.contains("spezi") || normalized.contains("spice") { return "🌶️" } // 💡 NUOVO: spezie generiche
        if normalized.contains("pepe") || normalized.contains("pepper") { return "🌶️" }
        if normalized.contains("cannella") || normalized.contains("cinnamon") { return "🥤" }
        if normalized.contains("chiodi") || normalized.contains("clove") { return "✨" }
        if normalized.contains("menta") || normalized.contains("mint") { return "🌿" }
        if normalized.contains("timo") || normalized.contains("thyme") { return "🌿" }
        if normalized.contains("rosmarino") || normalized.contains("rosemary") { return "🌿" }
        if normalized.contains("salvia") || normalized.contains("sage") { return "🍃" }
        if normalized.contains("erbace") || normalized.contains("herbal") { return "🌿" }
        
        // Legno & Tostato
        if normalized.contains("vaniglia") || normalized.contains("vanilla") { return "🍮" }
        if normalized.contains("tostato") || normalized.contains("toasted") { return "🍞" }
        if normalized.contains("caffè") || normalized.contains("coffee") { return "☕" }
        if normalized.contains("cioccolat") || normalized.contains("chocolate") { return "🍫" }
        if normalized.contains("cacao") || normalized.contains("cocoa") { return "🍫" }
        if normalized.contains("legno") || normalized.contains("wood") { return "🌲" }
        if normalized.contains("quercia") || normalized.contains("oak") { return "🌳" }
        if normalized.contains("cedro") { return "🌲" }
        
        // Terre & Minerale
        if normalized.contains("mineral") { return "🪨" }
        if normalized.contains("pietra") || normalized.contains("stone") { return "🪨" }
        if normalized.contains("terr") || normalized.contains("earth") { return "🌍" }
        if normalized.contains("fungo") || normalized.contains("mushroom") { return "🍄" }
        if normalized.contains("tartufo") || normalized.contains("truffle") { return "⭐" }
        
        // Altri
        if normalized.contains("cuoio") || normalized.contains("leather") { return "🧥" }
        if normalized.contains("tabacco") || normalized.contains("tobacco") { return "🍃" }
        if normalized.contains("miele") || normalized.contains("honey") { return "🍯" }
        if normalized.contains("cereali") || normalized.contains("cereal") { return "🌾" }
        if normalized.contains("pane") || normalized.contains("bread") { return "🍞" }
        if normalized.contains("burro") || normalized.contains("butter") { return "🧈" }
        if normalized.contains("noci") || normalized.contains("nuts") { return "🥜" }
        if normalized.contains("mandorl") || normalized.contains("almond") { return "🥜" }
        
        // Default
        return "🍇" // Uva
    }
}

// MARK: - Expanded Detail View (per bottone "Più dettagli")

struct WineDetailExpandedView: View {
    let wine: WineCard
    let userQuery: String
    
    @State private var tastingNotes: String = ""
    @State private var isLoadingNotes: Bool = false
    @State private var similarWines: [WineCard] = []
    @State private var isLoadingSimilar: Bool = false
    
    private let api = APIClient()
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                
                // ❌ Profilo aromatico RIMOSSO per richiesta utente
                
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
                
                // ❌ SENTORI RIMOSSI per richiesta utente
                
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
                
                // ✅ SEZIONE 1: VALUTAZIONI (punteggi numerici)
                if wine.quality != nil || wine.balance != nil || wine.persistence != nil {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Valutazioni")
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
                        }
                    }
                    .padding()
                    .background(AppColors.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                // ✅ SEZIONE 2: PROFILO ORGANOLETTICO (valori testuali)
                if wine.intensity != nil || wine.tannins != nil || wine.acidity != nil || wine.color_detail != nil {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Profilo organolettico")
                            .font(.headline)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            if let i = wine.intensity {
                                CharacteristicRow(label: "Intensità", value: WineLocalizer.intensity(i))
                            }
                            if let t = wine.tannins {
                                CharacteristicRow(label: "Tannicità", value: WineLocalizer.tannins(t))
                            }
                            if let a = wine.acidity {
                                CharacteristicRow(label: "Acidità", value: WineLocalizer.acidity(a))
                            }
                            if let c = wine.color_detail {
                                CharacteristicRow(label: "Colore", value: WineLocalizer.color(c))
                            }
                        }
                    }
                    .padding()
                    .background(AppColors.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                // 📏 GRAFICO CARATTERISTICHE (barre orizzontali)
                VStack(alignment: .leading, spacing: 12) {
                    Text("Profilo sensoriale")
                        .font(.headline)
                    
                    VStack(spacing: 10) {
                        // Intensità
                        if let intensity = wine.intensity {
                            CharacteristicBar(label: "INTENSO", value: barValue(intensity))
                        }
                        
                        // Tannicità
                        if let tannins = wine.tannins {
                            CharacteristicBar(label: "TANNICO", value: barValue(tannins))
                        }
                        
                        // Morbido (inverso di tannins)
                        if let tannins = wine.tannins {
                            CharacteristicBar(label: "MORBIDO", value: 1.0 - barValue(tannins))
                        }
                        
                        // Acidità (Fresco)
                        if let acidity = wine.acidity {
                            CharacteristicBar(label: "FRESCO", value: barValue(acidity))
                        }
                        
                        // Sapido (da freshness)
                        if let freshness = wine.freshness {
                            CharacteristicBar(label: "SAPIDO", value: barValue(freshness))
                        }
                        
                        // Frizzante
                        if let sparkling = wine.sparkling {
                            CharacteristicBar(label: "FRIZZANTE", value: sparklingBarValue(sparkling))
                        }
                        
                        // Dolcezza
                        if let sweetness = wine.sweetness {
                            CharacteristicBar(label: "DOLCE", value: sweetnessBarValue(sweetness))
                        }
                        
                        // Alcolico
                        if let alcohol = wine.alcohol_level {
                            CharacteristicBar(label: "ALCOLICO", value: barValue(alcohol))
                        }
                        
                        // Persistente
                        if let persistence = wine.persistence {
                            CharacteristicBar(label: "PERSISTENTE", value: persistenceBarValue(persistence))
                        }
                        
                        // Corposo/Cospicuo (da intensity)
                        if let intensity = wine.intensity {
                            CharacteristicBar(label: "CORPOSO", value: barValue(intensity))
                        }
                    }
                }
                .padding()
                .background(AppColors.cardBackground)
                .clipShape(RoundedRectangle(cornerRadius: 16))
                
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
        .navigationTitle("Dettagli completi")
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
    
    // 📏 Helper per convertire valori low/medium/high in 0.0...1.0
    private func barValue(_ value: String) -> Double {
        switch value.lowercased() {
        case "low", "bassa", "basso": return 0.33
        case "medium", "media", "medio": return 0.66
        case "medium_plus", "media-alta": return 0.83
        case "high", "alta", "alto": return 1.0
        default: return 0.5
        }
    }
    
    private func sparklingBarValue(_ value: String) -> Double {
        switch value.lowercased() {
        case "fermo": return 0.0
        case "frizzante": return 0.66
        case "spumante": return 1.0
        default: return 0.0
        }
    }
    
    private func sweetnessBarValue(_ value: String) -> Double {
        switch value.lowercased() {
        case "secco", "dry": return 0.0
        case "abboccato", "off-dry": return 0.5
        case "dolce", "sweet": return 1.0
        default: return 0.0
        }
    }
    
    private func persistenceBarValue(_ value: String) -> Double {
        // Persistence è un valore numerico 0-5, convertiamo in 0.0-1.0
        if let numeric = Double(value) {
            return min(max(numeric / 5.0, 0.0), 1.0)
        }
        return 0.5
    }
    
    // 🍒 Mappa aromi -> emoji
    private func aromaEmoji(for aroma: String) -> String {
        let normalized = aroma.lowercased().trimmingCharacters(in: .whitespaces)
        
        // Frutta
        if normalized.contains("ciliegi") || normalized.contains("cherry") { return "🍒" }
        if normalized.contains("fragol") || normalized.contains("strawberry") { return "🍓" }
        if normalized.contains("lampon") || normalized.contains("raspberry") { return "🥝" }
        if normalized.contains("mora") || normalized.contains("blackberry") { return "🫛" }
        if normalized.contains("mirtill") || normalized.contains("blueberry") { return "🫐" }
        if normalized.contains("prun") || normalized.contains("plum") { return "🍑" }
        if normalized.contains("ribes") { return "🍇" }
        if normalized.contains("mela") || normalized.contains("apple") { return "🍏" }
        if normalized.contains("pera") || normalized.contains("pear") { return "🍐" }
        if normalized.contains("pesca") || normalized.contains("peach") { return "🍑" }
        if normalized.contains("albicocc") || normalized.contains("apricot") { return "🥝" }
        if normalized.contains("agrumi") || normalized.contains("citrus") { return "🍋" }
        if normalized.contains("limon") || normalized.contains("lemon") { return "🍋" }
        
        // Fiori
        if normalized.contains("rosa") || normalized.contains("rose") { return "🌹" }
        if normalized.contains("viola") || normalized.contains("violet") { return "🌸" }
        if normalized.contains("gelsomino") || normalized.contains("jasmine") { return "🌼" }
        if normalized.contains("fiori") || normalized.contains("floral") { return "🌸" }
        
        // Spezie & Erbe
        if normalized.contains("pepe") || normalized.contains("pepper") { return "🌶️" }
        if normalized.contains("cannella") || normalized.contains("cinnamon") { return "🥤" }
        if normalized.contains("chiodi") || normalized.contains("clove") { return "✨" }
        if normalized.contains("menta") || normalized.contains("mint") { return "🌿" }
        if normalized.contains("timo") || normalized.contains("thyme") { return "🌿" }
        if normalized.contains("rosmarino") || normalized.contains("rosemary") { return "🌿" }
        if normalized.contains("salvia") || normalized.contains("sage") { return "🍃" }
        if normalized.contains("erbace") || normalized.contains("herbal") { return "🌿" }
        
        // Legno & Tostato
        if normalized.contains("vaniglia") || normalized.contains("vanilla") { return "🍮" }
        if normalized.contains("tostato") || normalized.contains("toasted") { return "🍞" }
        if normalized.contains("caffè") || normalized.contains("coffee") { return "☕" }
        if normalized.contains("cioccolat") || normalized.contains("chocolate") { return "🍫" }
        if normalized.contains("cacao") || normalized.contains("cocoa") { return "🍫" }
        if normalized.contains("legno") || normalized.contains("wood") { return "🌲" }
        if normalized.contains("quercia") || normalized.contains("oak") { return "🌳" }
        if normalized.contains("cedro") { return "🌲" }
        
        // Terre & Minerale
        if normalized.contains("mineral") { return "🪨" }
        if normalized.contains("pietra") || normalized.contains("stone") { return "🪨" }
        if normalized.contains("terr") || normalized.contains("earth") { return "🌍" }
        if normalized.contains("fungo") || normalized.contains("mushroom") { return "🍄" }
        if normalized.contains("tartufo") || normalized.contains("truffle") { return "⭐" }
        
        // Altri
        if normalized.contains("cuoio") || normalized.contains("leather") { return "🧥" }
        if normalized.contains("tabacco") || normalized.contains("tobacco") { return "🍃" }
        if normalized.contains("miele") || normalized.contains("honey") { return "🍯" }
        if normalized.contains("cereali") || normalized.contains("cereal") { return "🌾" }
        if normalized.contains("pane") || normalized.contains("bread") { return "🍞" }
        if normalized.contains("burro") || normalized.contains("butter") { return "🧈" }
        if normalized.contains("noci") || normalized.contains("nuts") { return "🥜" }
        if normalized.contains("mandorl") || normalized.contains("almond") { return "🥜" }
        
        // Default
        return "🍇" // Uva
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

// 📏 Barra orizzontale per caratteristiche (MAX 50% width)
struct CharacteristicBar: View {
    let label: String
    let value: Double // 0.0 ... 1.0
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            
            GeometryReader { geo in
                // ✅ FIX: MAX 50% della larghezza totale
                let maxBarWidth = geo.size.width * 0.5
                
                ZStack(alignment: .leading) {
                    // Background (full 50% width)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.15))
                        .frame(width: maxBarWidth)
                    
                    // Foreground bar (value * 50%)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(AppColors.primaryWine)
                        .frame(width: maxBarWidth * value)
                }
            }
            .frame(height: 8)
        }
    }
}

// MARK: - FlowLayout

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
