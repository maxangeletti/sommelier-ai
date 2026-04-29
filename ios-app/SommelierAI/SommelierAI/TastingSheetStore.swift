//
//  TastingSheetStore.swift
//  SommelierAI
//
//  Store per gestione schede degustazione
//

import Foundation
import Combine

class TastingSheetStore: ObservableObject {
    
    @Published var tastingSheets: [TastingSheet] = []
    
    private let tastingSheetsKey = "tastingSheets"
    
    init() {
        loadTastingSheets()
    }
    
    // MARK: - Add / Update
    
    func save(_ sheet: TastingSheet) {
        if let index = tastingSheets.firstIndex(where: { $0.id == sheet.id }) {
            tastingSheets[index] = sheet
        } else {
            tastingSheets.insert(sheet, at: 0)
        }
        persist()
    }
    
    // MARK: - Delete
    
    func delete(_ sheet: TastingSheet) {
        tastingSheets.removeAll { $0.id == sheet.id }
        persist()
    }
    
    func deleteAll() {
        tastingSheets.removeAll()
        persist()
    }
    
    // MARK: - Query
    
    func sheet(for wineId: String) -> TastingSheet? {
        tastingSheets.first { $0.wineId == wineId }
    }
    
    func sheets(for wineId: String) -> [TastingSheet] {
        tastingSheets.filter { $0.wineId == wineId }
    }
    
    var completedSheets: [TastingSheet] {
        tastingSheets.filter { $0.isComplete }
    }
    
    var incompleteSheetsCount: Int {
        tastingSheets.filter { !$0.isComplete }.count
    }
    
    // MARK: - Persistence
    
    private func loadTastingSheets() {
        guard let data = UserDefaults.standard.data(forKey: tastingSheetsKey),
              let decoded = try? JSONDecoder().decode([TastingSheet].self, from: data) else {
            tastingSheets = []
            return
        }
        tastingSheets = decoded
    }
    
    private func persist() {
        guard let encoded = try? JSONEncoder().encode(tastingSheets) else { return }
        UserDefaults.standard.set(encoded, forKey: tastingSheetsKey)
    }
}
