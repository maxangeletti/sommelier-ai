//
//  ContentView.swift
//  SommelierAI
//

import SwiftUI

struct ContentView: View {
    // ✅ Traccia se l'utente ha visto la schermata di onboarding
    @AppStorage("hasSeenWelcome") private var hasSeenWelcome = false
    
    // ✅ Query iniziale dalla WelcomeView
    @State private var initialSearchQuery = ""
    
    var body: some View {
        Group {
            if hasSeenWelcome {
                // Schermata principale con tabs
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
            } else {
                // ✅ Schermata di onboarding
                WelcomeView(
                    hasSeenWelcome: $hasSeenWelcome,
                    searchQuery: $initialSearchQuery
                )
            }
        }
        .onAppear {
            // 🔧 DEBUG: Decomment per resettare e vedere welcome
            hasSeenWelcome = false
        }
    }
}
