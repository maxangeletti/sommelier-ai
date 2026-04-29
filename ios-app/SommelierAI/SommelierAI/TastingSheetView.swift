//
//  TastingSheetView.swift
//  SommelierAI
//
//  Scheda degustazione professionale completa
//

import SwiftUI

struct TastingSheetView: View {
    
    let wine: WineCard
    @EnvironmentObject private var tastingSheetStore: TastingSheetStore
    @EnvironmentObject private var favoritesStore: FavoritesStore
    @Environment(\.dismiss) private var dismiss
    
    @State private var sheet: TastingSheet
    @State private var expandedSection: TastingSection? = .visual
    
    private enum TastingSection: String, CaseIterable {
        case visual = "👁️ Esame Visivo"
        case olfactory = "👃 Esame Olfattivo"
        case gustatory = "👅 Esame Gustativo"
        case final = "📝 Giudizio Finale"
    }
    
    init(wine: WineCard) {
        self.wine = wine
        _sheet = State(initialValue: TastingSheet(
            wineId: wine.id,
            wineName: wine.name
        ))
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                
                // Header Vino
                wineHeader
                
                Divider()
                
                // Sezioni Degustazione
                VStack(spacing: 12) {
                    visualSection
                    olfactorySection
                    gustatorySection
                    finalSection
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 20)
                
                // Progress Bar
                progressBar
                
                // Bottoni Azione
                actionButtons
                    .padding(.top, 20)
                    .padding(.bottom, 40)
            }
        }
        .background(AppColors.backgroundPrimary)
        .navigationTitle("Scheda Degustazione")
        .navigationBarTitleDisplayMode(.inline)
    }
    
    // MARK: - Wine Header
    
    private var wineHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(wine.name)
                .font(.title2.weight(.bold))
                .foregroundColor(AppColors.textPrimary)
            
            if let producer = wine.producer {
                Text(producer)
                    .font(.subheadline)
                    .foregroundColor(AppColors.textSecondary)
            }
            
            if let vintage = wine.vintage {
                Text("Annata \(vintage)")
                    .font(.caption)
                    .foregroundColor(AppColors.textMuted)
            }
            
            Text("Data: \(formattedDate(sheet.date))")
                .font(.caption)
                .foregroundColor(AppColors.textMuted)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(AppColors.backgroundSecondary)
    }
    
    // MARK: - Visual Section
    
    private var visualSection: some View {
        ExpandableSection(
            title: TastingSection.visual.rawValue,
            isExpanded: Binding(
                get: { expandedSection == .visual },
                set: { if $0 { expandedSection = .visual } else { expandedSection = nil } }
            )
        ) {
            VStack(spacing: 16) {
                
                // Colore
                VStack(alignment: .leading, spacing: 8) {
                    Text("Colore")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(AppColors.textPrimary)
                    
                    TextField("Es: Rosso rubino, Giallo paglierino...", text: Binding(
                        get: { sheet.visualColor ?? "" },
                        set: { sheet.visualColor = $0.isEmpty ? nil : $0 }
                    ))
                    .textFieldStyle(.roundedBorder)
                }
                
                // Intensità
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Intensità")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(AppColors.textPrimary)
                        
                        Spacer()
                        
                        if let intensity = sheet.visualIntensity {
                            Text(intensityLabel(intensity))
                                .font(.caption)
                                .foregroundColor(AppColors.textSecondary)
                        }
                    }
                    
                    StarRating(rating: Binding(
                        get: { sheet.visualIntensity ?? 0 },
                        set: { sheet.visualIntensity = $0 }
                    ))
                }
                
                // Limpidezza
                VStack(alignment: .leading, spacing: 8) {
                    Text("Limpidezza")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(AppColors.textPrimary)
                    
                    Picker("Limpidezza", selection: Binding(
                        get: { sheet.visualClarity ?? "" },
                        set: { sheet.visualClarity = $0.isEmpty ? nil : $0 }
                    )) {
                        Text("Seleziona").tag("")
                        Text("Limpido").tag("limpido")
                        Text("Velato").tag("velato")
                        Text("Torbido").tag("torbido")
                    }
                    .pickerStyle(.segmented)
                }
            }
        }
    }
    
    // MARK: - Olfactory Section
    
    private var olfactorySection: some View {
        ExpandableSection(
            title: TastingSection.olfactory.rawValue,
            isExpanded: Binding(
                get: { expandedSection == .olfactory },
                set: { if $0 { expandedSection = .olfactory } else { expandedSection = nil } }
            )
        ) {
            VStack(spacing: 16) {
                
                // Aromi (Multi-select)
                VStack(alignment: .leading, spacing: 8) {
                    Text("Aromi Rilevati")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(AppColors.textPrimary)
                    
                    AromaSelector(selectedAromas: $sheet.olfactoryAromas)
                }
                
                // Intensità
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Intensità")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(AppColors.textPrimary)
                        
                        Spacer()
                        
                        if let intensity = sheet.olfactoryIntensity {
                            Text(intensityLabel(intensity))
                                .font(.caption)
                                .foregroundColor(AppColors.textSecondary)
                        }
                    }
                    
                    StarRating(rating: Binding(
                        get: { sheet.olfactoryIntensity ?? 0 },
                        set: { sheet.olfactoryIntensity = $0 }
                    ))
                }
                
                // Complessità
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Complessità")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(AppColors.textPrimary)
                        
                        Spacer()
                        
                        if let complexity = sheet.olfactoryComplexity {
                            Text(complexityLabel(complexity))
                                .font(.caption)
                                .foregroundColor(AppColors.textSecondary)
                        }
                    }
                    
                    StarRating(rating: Binding(
                        get: { sheet.olfactoryComplexity ?? 0 },
                        set: { sheet.olfactoryComplexity = $0 }
                    ))
                }
                
                // Note libere
                VStack(alignment: .leading, spacing: 8) {
                    Text("Note Olfattive")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(AppColors.textPrimary)
                    
                    TextEditor(text: Binding(
                        get: { sheet.olfactoryNotes ?? "" },
                        set: { sheet.olfactoryNotes = $0.isEmpty ? nil : $0 }
                    ))
                    .frame(height: 80)
                    .padding(8)
                    .background(AppColors.backgroundSecondary)
                    .cornerRadius(8)
                }
            }
        }
    }
    
    // MARK: - Gustatory Section
    
    private var gustatorySection: some View {
        ExpandableSection(
            title: TastingSection.gustatory.rawValue,
            isExpanded: Binding(
                get: { expandedSection == .gustatory },
                set: { if $0 { expandedSection = .gustatory } else { expandedSection = nil } }
            )
        ) {
            VStack(spacing: 16) {
                
                // Dolcezza
                ratingRow(
                    title: "Dolcezza",
                    rating: Binding(
                        get: { sheet.gustatorySweetness ?? 0 },
                        set: { sheet.gustatorySweetness = $0 }
                    ),
                    labels: ["Secco", "Abboccato", "Amabile", "Dolce", "Molto Dolce"]
                )
                
                // Acidità
                ratingRow(
                    title: "Acidità",
                    rating: Binding(
                        get: { sheet.gustatoryAcidity ?? 0 },
                        set: { sheet.gustatoryAcidity = $0 }
                    ),
                    labels: ["Piatta", "Scarsa", "Equilibrata", "Vivace", "Molto Vivace"]
                )
                
                // Tannini (solo per rossi)
                if isRedWine {
                    ratingRow(
                        title: "Tannini",
                        rating: Binding(
                            get: { sheet.gustatoryTannins ?? 0 },
                            set: { sheet.gustatoryTannins = $0 }
                        ),
                        labels: ["Assenti", "Morbidi", "Presenti", "Marcati", "Molto Marcati"]
                    )
                }
                
                // Corpo
                ratingRow(
                    title: "Corpo",
                    rating: Binding(
                        get: { sheet.gustatoryBody ?? 0 },
                        set: { sheet.gustatoryBody = $0 }
                    ),
                    labels: ["Leggero", "Snello", "Medio", "Robusto", "Pieno"]
                )
                
                // Equilibrio
                ratingRow(
                    title: "Equilibrio",
                    rating: Binding(
                        get: { sheet.gustatoryBalance ?? 0 },
                        set: { sheet.gustatoryBalance = $0 }
                    ),
                    labels: ["Scarso", "Sufficiente", "Buono", "Ottimo", "Eccellente"]
                )
                
                // Persistenza
                ratingRow(
                    title: "Persistenza",
                    rating: Binding(
                        get: { sheet.gustatoryPersistence ?? 0 },
                        set: { sheet.gustatoryPersistence = $0 }
                    ),
                    labels: ["Corta", "Media", "Lunga", "Molto Lunga", "Infinita"]
                )
                
                // Note libere
                VStack(alignment: .leading, spacing: 8) {
                    Text("Note Gustative")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(AppColors.textPrimary)
                    
                    TextEditor(text: Binding(
                        get: { sheet.gustatoryNotes ?? "" },
                        set: { sheet.gustatoryNotes = $0.isEmpty ? nil : $0 }
                    ))
                    .frame(height: 80)
                    .padding(8)
                    .background(AppColors.backgroundSecondary)
                    .cornerRadius(8)
                }
            }
        }
    }
    
    // MARK: - Final Section
    
    private var finalSection: some View {
        ExpandableSection(
            title: TastingSection.final.rawValue,
            isExpanded: Binding(
                get: { expandedSection == .final },
                set: { if $0 { expandedSection = .final } else { expandedSection = nil } }
            )
        ) {
            VStack(spacing: 16) {
                
                // Rating Finale
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Valutazione Complessiva")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(AppColors.textPrimary)
                        
                        Spacer()
                        
                        if let rating = sheet.finalRating {
                            Text(ratingLabel(rating))
                                .font(.caption.weight(.semibold))
                                .foregroundColor(AppColors.accentWine)
                        }
                    }
                    
                    StarRating(rating: Binding(
                        get: { sheet.finalRating ?? 0 },
                        set: { sheet.finalRating = $0 }
                    ), size: 36)
                }
                
                // Note Finali
                VStack(alignment: .leading, spacing: 8) {
                    Text("Note Conclusive")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(AppColors.textPrimary)
                    
                    TextEditor(text: Binding(
                        get: { sheet.finalNotes ?? "" },
                        set: { sheet.finalNotes = $0.isEmpty ? nil : $0 }
                    ))
                    .frame(height: 100)
                    .padding(8)
                    .background(AppColors.backgroundSecondary)
                    .cornerRadius(8)
                }
            }
        }
    }
    
    // MARK: - Progress Bar
    
    private var progressBar: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Completamento")
                    .font(.caption)
                    .foregroundColor(AppColors.textSecondary)
                
                Spacer()
                
                Text("\(sheet.completionPercentage)%")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(AppColors.primaryWine)
            }
            .padding(.horizontal, 16)
            
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 6)
                    
                    Capsule()
                        .fill(AppColors.primaryWine)
                        .frame(width: geo.size.width * CGFloat(sheet.completionPercentage) / 100, height: 6)
                }
            }
            .frame(height: 6)
            .padding(.horizontal, 16)
        }
    }
    
    // MARK: - Action Buttons
    
    private var actionButtons: some View {
        VStack(spacing: 12) {
            // Salva
            Button(action: {
                tastingSheetStore.save(sheet)
                dismiss()
            }) {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                    Text("Salva Scheda")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .foregroundColor(.white)
                .background(AppColors.primaryWine)
                .cornerRadius(12)
            }
            
            // Salva + Aggiungi a Preferiti
            Button(action: {
                tastingSheetStore.save(sheet)
                if !favoritesStore.isFavorite(wine) {
                    favoritesStore.toggle(wine)
                }
                dismiss()
            }) {
                HStack {
                    Image(systemName: "heart.fill")
                    Text("Salva e Aggiungi a Preferiti")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .foregroundColor(AppColors.primaryWine)
                .background(AppColors.primaryWine.opacity(0.1))
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(AppColors.primaryWine, lineWidth: 2)
                )
            }
        }
        .padding(.horizontal, 16)
    }
    
    // MARK: - Helpers
    
    private var isRedWine: Bool {
        wine.color?.lowercased().contains("red") ?? false ||
        wine.color?.lowercased().contains("rosso") ?? false
    }
    
    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        formatter.locale = Locale(identifier: "it_IT")
        return formatter.string(from: date)
    }
    
    private func intensityLabel(_ value: Int) -> String {
        ["Molto Bassa", "Bassa", "Media", "Alta", "Molto Alta"][max(0, min(4, value - 1))]
    }
    
    private func complexityLabel(_ value: Int) -> String {
        ["Semplice", "Poco Complesso", "Moderato", "Complesso", "Molto Complesso"][max(0, min(4, value - 1))]
    }
    
    private func ratingLabel(_ value: Int) -> String {
        ["Scarso", "Sufficiente", "Buono", "Ottimo", "Eccellente"][max(0, min(4, value - 1))]
    }
    
    @ViewBuilder
    private func ratingRow(title: String, rating: Binding<Int>, labels: [String]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(AppColors.textPrimary)
                
                Spacer()
                
                if rating.wrappedValue > 0, rating.wrappedValue <= labels.count {
                    Text(labels[rating.wrappedValue - 1])
                        .font(.caption)
                        .foregroundColor(AppColors.textSecondary)
                }
            }
            
            StarRating(rating: rating)
        }
    }
}

// MARK: - Supporting Views

private struct ExpandableSection<Content: View>: View {
    let title: String
    @Binding var isExpanded: Bool
    let content: () -> Content
    
    var body: some View {
        VStack(spacing: 0) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(AppColors.textPrimary)
                    
                    Spacer()
                    
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundColor(AppColors.textSecondary)
                }
                .padding(.vertical, 14)
                .padding(.horizontal, 16)
                .background(AppColors.backgroundSecondary)
                .cornerRadius(12)
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                VStack(spacing: 0) {
                    content()
                        .padding(16)
                }
                .background(AppColors.cardBackground)
                .cornerRadius(12)
                .padding(.top, 8)
            }
        }
    }
}

private struct StarRating: View {
    @Binding var rating: Int
    var size: CGFloat = 28
    
    var body: some View {
        HStack(spacing: 8) {
            ForEach(1...5, id: \.self) { index in
                Button(action: {
                    rating = index
                }) {
                    Image(systemName: index <= rating ? "star.fill" : "star")
                        .font(.system(size: size))
                        .foregroundColor(index <= rating ? AppColors.accentWine : Color.gray.opacity(0.3))
                }
                .buttonStyle(.plain)
            }
        }
    }
}

private struct AromaSelector: View {
    @Binding var selectedAromas: [String]
    
    private let availableAromas = [
        "🍇 Frutti rossi", "🍊 Agrumi", "🍎 Frutti bianchi",
        "🌸 Fiori", "🌿 Erbe", "🌶️ Spezie",
        "🍂 Note terrose", "🪵 Legno", "☕ Tostato",
        "🍯 Miele", "🧈 Burro", "🥜 Frutta secca"
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            FlowLayout(spacing: 8) {
                ForEach(availableAromas, id: \.self) { aroma in
                    Button(action: {
                        if selectedAromas.contains(aroma) {
                            selectedAromas.removeAll { $0 == aroma }
                        } else {
                            selectedAromas.append(aroma)
                        }
                    }) {
                        Text(aroma)
                            .font(.caption)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                selectedAromas.contains(aroma)
                                ? AppColors.primaryWine.opacity(0.2)
                                : AppColors.backgroundSecondary
                            )
                            .foregroundColor(
                                selectedAromas.contains(aroma)
                                ? AppColors.primaryWine
                                : AppColors.textPrimary
                            )
                            .cornerRadius(16)
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .stroke(
                                        selectedAromas.contains(aroma)
                                        ? AppColors.primaryWine
                                        : Color.gray.opacity(0.3),
                                        lineWidth: 1
                                    )
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}
