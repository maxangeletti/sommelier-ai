//
//  SearchHistoryStore.swift
//  SommelierAI
//
//  Gestisce storico ricerche recenti
//

import Foundation
import Combine   // ✅ REQUIRED for ObservableObject

class SearchHistoryStore: ObservableObject {
    
    private let maxHistory = 10
    private let historyKey = "searchHistory"
    
    @Published var recentSearches: [String] = []
    
    init() {
        loadHistory()
    }
    
    /// Aggiunge una ricerca allo storico
    func add(_ query: String) {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        // Rimuovi duplicati e aggiungi in cima
        recentSearches.removeAll { $0.lowercased() == trimmed.lowercased() }
        recentSearches.insert(trimmed, at: 0)
        
        // Mantieni solo le ultime N
        if recentSearches.count > maxHistory {
            recentSearches = Array(recentSearches.prefix(maxHistory))
        }
        
        saveHistory()
    }
    
    /// Rimuove una ricerca dallo storico
    func remove(_ query: String) {
        recentSearches.removeAll { $0.lowercased() == query.lowercased() }
        saveHistory()
    }
    
    /// Pulisce tutto lo storico
    func clear() {
        recentSearches.removeAll()
        saveHistory()
    }
    
    // MARK: - Persistence
    
    private func loadHistory() {
        if let saved = UserDefaults.standard.stringArray(forKey: historyKey) {
            recentSearches = saved
        }
    }
    
    private func saveHistory() {
        UserDefaults.standard.set(recentSearches, forKey: historyKey)
    }
}
