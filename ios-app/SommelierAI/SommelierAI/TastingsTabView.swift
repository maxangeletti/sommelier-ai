//
//  TastingsTabView.swift
//  SommelierAI
//
//  Tab per visualizzare tutte le schede degustazione salvate
//

import SwiftUI

struct TastingsTabView: View {
    
    @EnvironmentObject private var tastingSheetStore: TastingSheetStore
    @EnvironmentObject private var favoritesStore: FavoritesStore
    
    @State private var sortBy: TastingSortMode = .dateDesc
    @State private var showingDeleteAllAlert = false
    
    enum TastingSortMode: String, CaseIterable {
        case dateDesc = "Più recenti"
        case dateAsc = "Più vecchie"
        case ratingDesc = "Rating più alto"
        case nameAsc = "Nome A-Z"
        
        func sort(_ sheets: [TastingSheet]) -> [TastingSheet] {
            switch self {
            case .dateDesc:
                return sheets.sorted { $0.date > $1.date }
            case .dateAsc:
                return sheets.sorted { $0.date < $1.date }
            case .ratingDesc:
                return sheets.sorted { ($0.finalRating ?? 0) > ($1.finalRating ?? 0) }
            case .nameAsc:
                return sheets.sorted { $0.wineName < $1.wineName }
            }
        }
    }
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                
                // Header con sorting e delete
                if !tastingSheetStore.tastingSheets.isEmpty {
                    HStack {
                        Spacer()
                        
                        Menu {
                            Picker("Ordina per", selection: $sortBy) {
                                ForEach(TastingSortMode.allCases, id: \.self) { mode in
                                    Text(mode.rawValue).tag(mode)
                                }
                            }
                            .pickerStyle(.inline)
                        } label: {
                            Image(systemName: "arrow.up.arrow.down")
                                .foregroundColor(AppColors.textSecondary)
                        }
                        
                        Button(role: .destructive) {
                            showingDeleteAllAlert = true
                        } label: {
                            Image(systemName: "trash")
                                .foregroundColor(AppColors.accentWine)
                        }
                    }
                    .padding(.horizontal, Spacing.screenEdges)
                    .padding(.vertical, Spacing.md)
                    .background(AppColors.backgroundSecondary)
                    
                    Divider()
                }
                
                // Lista degustazioni
                if tastingSheetStore.tastingSheets.isEmpty {
                    emptyState
                } else {
                    tastingsList
                }
            }
            .background(AppColors.backgroundPrimary)
            .navigationTitle("Degustazioni")
            .navigationBarTitleDisplayMode(.large)
            .alert("Elimina Tutte", isPresented: $showingDeleteAllAlert) {
                Button("Annulla", role: .cancel) {}
                Button("Elimina", role: .destructive) {
                    tastingSheetStore.deleteAll()
                }
            } message: {
                Text("Vuoi eliminare tutte le schede degustazione? Questa azione non può essere annullata.")
            }
        }
    }
    
    // MARK: - Empty State
    
    private var emptyState: some View {
        VStack(spacing: 24) {
            Spacer()
            
            Image(systemName: "list.clipboard")
                .font(.system(size: 64))
                .foregroundColor(AppColors.textMuted.opacity(0.5))
            
            VStack(spacing: 8) {
                Text("Nessuna Degustazione")
                    .font(.title2.weight(.semibold))
                    .foregroundColor(AppColors.textPrimary)
                
                Text("Le schede degustazione che compili appariranno qui")
                    .font(.subheadline)
                    .foregroundColor(AppColors.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
            
            Spacer()
            
            Text("💡 Cerca un vino, apri i dettagli e tap il bottone 📋 per creare una scheda degustazione")
                .font(.caption)
                .foregroundColor(AppColors.textMuted)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
                .padding(.bottom, 40)
        }
    }
    
    // MARK: - List
    
    private var tastingsList: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                ForEach(sortBy.sort(tastingSheetStore.tastingSheets)) { sheet in
                    tastingRow(sheet)
                        .padding(.horizontal, Spacing.screenEdges)
                }
            }
            .padding(.vertical, Spacing.md)
        }
    }
    
    // MARK: - Row
    
    private func tastingRow(_ sheet: TastingSheet) -> some View {
        NavigationLink {
            TastingSheetDetailView(sheet: sheet)
        } label: {
            VStack(alignment: .leading, spacing: Spacing.sm) {
                
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(sheet.wineName)
                            .font(Typography.title3)
                            .foregroundColor(AppColors.textPrimary)
                            .lineLimit(2)
                        
                        Text(formattedDate(sheet.date))
                            .font(Typography.caption)
                            .foregroundColor(AppColors.textMuted)
                    }
                    
                    Spacer()
                    
                    // Rating stelle
                    if let rating = sheet.finalRating, rating > 0 {
                        HStack(spacing: 2) {
                            ForEach(1...5, id: \.self) { index in
                                Image(systemName: index <= rating ? "star.fill" : "star")
                                    .font(.system(size: 12))
                                    .foregroundColor(index <= rating ? AppColors.accentWine : Color.gray.opacity(0.3))
                            }
                        }
                    }
                }
                
                // Progress bar
                HStack(spacing: 8) {
                    Text("Completamento")
                        .font(.caption2)
                        .foregroundColor(AppColors.textSecondary)
                    
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 4)
                            
                            Capsule()
                                .fill(sheet.completionPercentage == 100 ? Color.green : AppColors.primaryWine)
                                .frame(width: geo.size.width * CGFloat(sheet.completionPercentage) / 100, height: 4)
                        }
                    }
                    .frame(height: 4)
                    
                    Text("\(sheet.completionPercentage)%")
                        .font(.caption2.weight(.semibold))
                        .foregroundColor(sheet.completionPercentage == 100 ? .green : AppColors.primaryWine)
                        .frame(width: 35, alignment: .trailing)
                }
                
                // Note preview (se presenti)
                if let notes = sheet.finalNotes, !notes.isEmpty {
                    Text(notes)
                        .font(Typography.caption)
                        .foregroundColor(AppColors.textSecondary)
                        .lineLimit(2)
                        .padding(.top, 4)
                }
            }
            .padding(Spacing.cardPadding)
            .background(AppColors.cardBackground)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(AppColors.borderLight, lineWidth: 0.5)
            )
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(role: .destructive) {
                tastingSheetStore.delete(sheet)
            } label: {
                Label("Elimina", systemImage: "trash")
            }
        }
    }
    
    // MARK: - Helpers
    
    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        formatter.locale = Locale(identifier: "it_IT")
        return formatter.string(from: date)
    }
}

// MARK: - Detail View (Read-Only)

struct TastingSheetDetailView: View {
    
    let sheet: TastingSheet
    @EnvironmentObject private var tastingSheetStore: TastingSheetStore
    @EnvironmentObject private var favoritesStore: FavoritesStore
    @Environment(\.dismiss) private var dismiss
    
    @State private var showDeleteAlert = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                
                // Header Vino
                VStack(alignment: .leading, spacing: 8) {
                    Text(sheet.wineName)
                        .font(.title.weight(.bold))
                        .foregroundColor(AppColors.textPrimary)
                    
                    Text("Degustato il \(formattedDate(sheet.date))")
                        .font(.subheadline)
                        .foregroundColor(AppColors.textSecondary)
                }
                .padding(.horizontal, 16)
                
                Divider()
                
                // Completamento
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Completamento")
                            .font(.caption)
                            .foregroundColor(AppColors.textSecondary)
                        
                        Spacer()
                        
                        Text("\(sheet.completionPercentage)%")
                            .font(.caption.weight(.semibold))
                            .foregroundColor(sheet.completionPercentage == 100 ? .green : AppColors.primaryWine)
                    }
                    
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 6)
                            
                            Capsule()
                                .fill(sheet.completionPercentage == 100 ? Color.green : AppColors.primaryWine)
                                .frame(width: geo.size.width * CGFloat(sheet.completionPercentage) / 100, height: 6)
                        }
                    }
                    .frame(height: 6)
                }
                .padding(.horizontal, 16)
                
                // Sezioni
                VStack(spacing: 16) {
                    if hasVisualData {
                        sectionCard(title: "👁️ Esame Visivo") {
                            visualSection
                        }
                    }
                    
                    if hasOlfactoryData {
                        sectionCard(title: "👃 Esame Olfattivo") {
                            olfactorySection
                        }
                    }
                    
                    if hasGustatoryData {
                        sectionCard(title: "👅 Esame Gustativo") {
                            gustatorySection
                        }
                    }
                    
                    if hasFinalData {
                        sectionCard(title: "📝 Giudizio Finale") {
                            finalSection
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 40)
            }
        }
        .background(AppColors.backgroundPrimary)
        .navigationTitle("Scheda Degustazione")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(role: .destructive) {
                    showDeleteAlert = true
                } label: {
                    Image(systemName: "trash")
                        .foregroundColor(AppColors.accentWine)
                }
            }
        }
        .alert("Elimina Scheda", isPresented: $showDeleteAlert) {
            Button("Annulla", role: .cancel) {}
            Button("Elimina", role: .destructive) {
                tastingSheetStore.delete(sheet)
                dismiss()
            }
        } message: {
            Text("Vuoi eliminare questa scheda degustazione? Questa azione non può essere annullata.")
        }
    }
    
    // MARK: - Section Card
    
    @ViewBuilder
    private func sectionCard<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
                .foregroundColor(AppColors.textPrimary)
            
            content()
        }
        .padding(16)
        .background(AppColors.cardBackground)
        .cornerRadius(12)
    }
    
    // MARK: - Visual Section
    
    private var visualSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let color = sheet.visualColor {
                dataRow(label: "Colore", value: color)
            }
            if let intensity = sheet.visualIntensity {
                dataRow(label: "Intensità", value: intensityLabel(intensity))
            }
            if let clarity = sheet.visualClarity {
                dataRow(label: "Limpidezza", value: clarity.capitalized)
            }
        }
    }
    
    // MARK: - Olfactory Section
    
    private var olfactorySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !sheet.olfactoryAromas.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Aromi")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(AppColors.textSecondary)
                    
                    FlowLayout(spacing: 6) {
                        ForEach(sheet.olfactoryAromas, id: \.self) { aroma in
                            Text(aroma)
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(AppColors.primaryWine.opacity(0.1))
                                .foregroundColor(AppColors.primaryWine)
                                .cornerRadius(12)
                        }
                    }
                }
            }
            if let intensity = sheet.olfactoryIntensity {
                dataRow(label: "Intensità", value: intensityLabel(intensity))
            }
            if let complexity = sheet.olfactoryComplexity {
                dataRow(label: "Complessità", value: complexityLabel(complexity))
            }
            if let notes = sheet.olfactoryNotes, !notes.isEmpty {
                dataRow(label: "Note", value: notes)
            }
        }
    }
    
    // MARK: - Gustatory Section
    
    private var gustatorySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let sweetness = sheet.gustatorySweetness {
                ratingRow(label: "Dolcezza", rating: sweetness, labels: ["Secco", "Abboccato", "Amabile", "Dolce", "Molto Dolce"])
            }
            if let acidity = sheet.gustatoryAcidity {
                ratingRow(label: "Acidità", rating: acidity, labels: ["Piatta", "Scarsa", "Equilibrata", "Vivace", "Molto Vivace"])
            }
            if let tannins = sheet.gustatoryTannins {
                ratingRow(label: "Tannini", rating: tannins, labels: ["Assenti", "Morbidi", "Presenti", "Marcati", "Molto Marcati"])
            }
            if let body = sheet.gustatoryBody {
                ratingRow(label: "Corpo", rating: body, labels: ["Leggero", "Snello", "Medio", "Robusto", "Pieno"])
            }
            if let balance = sheet.gustatoryBalance {
                ratingRow(label: "Equilibrio", rating: balance, labels: ["Scarso", "Sufficiente", "Buono", "Ottimo", "Eccellente"])
            }
            if let persistence = sheet.gustatoryPersistence {
                ratingRow(label: "Persistenza", rating: persistence, labels: ["Corta", "Media", "Lunga", "Molto Lunga", "Infinita"])
            }
            if let notes = sheet.gustatoryNotes, !notes.isEmpty {
                dataRow(label: "Note", value: notes)
            }
        }
    }
    
    // MARK: - Final Section
    
    private var finalSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let rating = sheet.finalRating {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Valutazione Complessiva")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(AppColors.textSecondary)
                    
                    HStack(spacing: 4) {
                        ForEach(1...5, id: \.self) { index in
                            Image(systemName: index <= rating ? "star.fill" : "star")
                                .font(.system(size: 20))
                                .foregroundColor(index <= rating ? AppColors.accentWine : Color.gray.opacity(0.3))
                        }
                        
                        Text(ratingLabel(rating))
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(AppColors.accentWine)
                            .padding(.leading, 8)
                    }
                }
            }
            
            if let notes = sheet.finalNotes, !notes.isEmpty {
                dataRow(label: "Note Conclusive", value: notes)
            }
        }
    }
    
    // MARK: - Helper Views
    
    private func dataRow(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundColor(AppColors.textSecondary)
            Text(value)
                .font(.subheadline)
                .foregroundColor(AppColors.textPrimary)
        }
    }
    
    private func ratingRow(label: String, rating: Int, labels: [String]) -> some View {
        HStack {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundColor(AppColors.textSecondary)
            
            Spacer()
            
            if rating > 0, rating <= labels.count {
                Text(labels[rating - 1])
                    .font(.caption)
                    .foregroundColor(AppColors.textPrimary)
            }
            
            HStack(spacing: 2) {
                ForEach(1...5, id: \.self) { index in
                    Image(systemName: index <= rating ? "star.fill" : "star")
                        .font(.system(size: 10))
                        .foregroundColor(index <= rating ? AppColors.accentWine : Color.gray.opacity(0.3))
                }
            }
        }
    }
    
    // MARK: - Helpers
    
    private var hasVisualData: Bool {
        sheet.visualColor != nil || sheet.visualIntensity != nil || sheet.visualClarity != nil
    }
    
    private var hasOlfactoryData: Bool {
        !sheet.olfactoryAromas.isEmpty || sheet.olfactoryIntensity != nil || sheet.olfactoryComplexity != nil || sheet.olfactoryNotes != nil
    }
    
    private var hasGustatoryData: Bool {
        sheet.gustatorySweetness != nil || sheet.gustatoryAcidity != nil || sheet.gustatoryTannins != nil ||
        sheet.gustatoryBody != nil || sheet.gustatoryBalance != nil || sheet.gustatoryPersistence != nil || sheet.gustatoryNotes != nil
    }
    
    private var hasFinalData: Bool {
        sheet.finalRating != nil || sheet.finalNotes != nil
    }
    
    private func formattedDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .long
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
}
