import SwiftUI
import UIKit

@main
struct SommelierAIApp: App {
    @StateObject private var favoritesStore = FavoritesStore()
    @StateObject private var tierStore = TierStore()
    @StateObject private var searchHistoryStore = SearchHistoryStore()
    @StateObject private var tastingSheetStore = TastingSheetStore()   // ✅ NEW

    init() {
        let bordeaux = UIColor(red: 0.45, green: 0.05, blue: 0.15, alpha: 1.0)

        let nav = UINavigationBarAppearance()
        nav.configureWithOpaqueBackground()
        nav.backgroundColor = .systemBackground
        nav.titleTextAttributes = [.foregroundColor: UIColor.label]
        nav.largeTitleTextAttributes = [.foregroundColor: UIColor.label]

        UINavigationBar.appearance().standardAppearance = nav
        UINavigationBar.appearance().scrollEdgeAppearance = nav
        UINavigationBar.appearance().compactAppearance = nav
        UINavigationBar.appearance().tintColor = bordeaux

        let tab = UITabBarAppearance()
        tab.configureWithOpaqueBackground()
        tab.backgroundColor = .systemBackground

        func apply(_ item: UITabBarItemAppearance) {
            item.selected.iconColor = bordeaux
            item.selected.titleTextAttributes = [.foregroundColor: bordeaux]
            item.normal.iconColor = .secondaryLabel
            item.normal.titleTextAttributes = [.foregroundColor: UIColor.secondaryLabel]
        }
        apply(tab.stackedLayoutAppearance)
        apply(tab.inlineLayoutAppearance)
        apply(tab.compactInlineLayoutAppearance)

        UITabBar.appearance().standardAppearance = tab
        if #available(iOS 15.0, *) {
            UITabBar.appearance().scrollEdgeAppearance = tab
        }
        UITabBar.appearance().tintColor = bordeaux
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .tint(Color(red: 0.45, green: 0.05, blue: 0.15))
                .environmentObject(favoritesStore)
                .environmentObject(tierStore)
                .environmentObject(searchHistoryStore)
                .environmentObject(tastingSheetStore)   // ✅ NEW
        }
    }
}
