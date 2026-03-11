//
//  ContentView.swift
//  SommelierAI
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            NavigationStack { ChatView() }
                .tabItem {
                    Image(systemName: "bubble.left.and.bubble.right")
                    Text("Chat")
                }

            NavigationStack { FavoritesTabView() }
                .tabItem {
                    Image(systemName: "heart.fill")
                    Text("Preferiti")
                }
        }
        // ✅ Forza colori corretti di navigation bar e tab bar
        .toolbarColorScheme(.light, for: .navigationBar)
        .toolbarColorScheme(.light, for: .tabBar)
    }
}
