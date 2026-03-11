//
//  FavoritesStore.swift
//  SommelierAI
//
//  PURPOSE
//  Local persistence and state manager for user favorite wines.
//
//  SCOPE
//  - Store and persist [WineCard]
//  - Manage sorting preference for favorites
//  - Use ChatDomain.SortMode.less() as single source of truth
//
//  OUT OF SCOPE
//  - Networking
//  - Domain model definitions
//  - UI rendering
//
//  RULES
//  - Sorting logic MUST delegate to SortMode.less()
//  - Do not duplicate comparator logic here
//

import Foundation
import Combine

@MainActor
final class FavoritesStore: ObservableObject {

    @Published private(set) var favorites: [WineCard] = []

    private var isLoading = false

    @Published var sort: ChatDomain.SortMode = .relevance {
        didSet {
            guard !isLoading else { return }
            saveSort()
        }
    }

    // MARK: - Sorted View

    var sortedFavorites: [WineCard] {
        favorites.sorted { sort.less($0, $1) }
    }

    // MARK: - Persistence

    private let storageKey = "sommelier_favorites_v1"
    private let sortKey = "sommelier_sort_favorites_v1"

    init() {
        isLoading = true
        load()
        loadSort()
        isLoading = false
    }

    func isFavorite(_ wine: WineCard) -> Bool {
        favorites.contains(where: { $0.id == wine.id })
    }

    func toggle(_ wine: WineCard) {
        if let idx = favorites.firstIndex(where: { $0.id == wine.id }) {
            favorites.remove(at: idx)
        } else {
            favorites.insert(wine, at: 0)
        }
        save()
    }

    func clear() {
        favorites.removeAll()
        save()
    }

    private func save() {
        if let data = try? JSONEncoder().encode(favorites) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let arr = try? JSONDecoder().decode([WineCard].self, from: data) else {
            favorites = []
            return
        }
        favorites = arr
    }

    private func saveSort() {
        UserDefaults.standard.set(sort.rawValue, forKey: sortKey)
    }

    private func loadSort() {
        guard let raw = UserDefaults.standard.string(forKey: sortKey),
              let s = ChatDomain.SortMode(rawValue: raw) else { return }
        sort = s
    }
}
