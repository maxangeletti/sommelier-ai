//
//  ContentView.swift
//  SommelierAI
//

import SwiftUI

struct ContentView: View {
    // ✅ Welcome ad OGNI avvio app (non salvato in UserDefaults)
    @State private var hasSeenWelcome = false
    
    // ✅ Query iniziale dalla WelcomeView
    @State private var initialSearchQuery = ""
    
    // ✅ Mostra splash screen iniziale
    @State private var showingSplash = true
    
    var body: some View {
        Group {
            if showingSplash {
                // ✅ SPLASH SCREEN
                LaunchScreenView()
                    .onAppear {
                        // Mostra per 2 secondi
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                            withAnimation {
                                showingSplash = false
                            }
                        }
                    }
            } else {
                // ✅ SEMPRE TabView - tab bar sempre visibile
                TabView {
                    // Tab Chat: WelcomeView o ChatView in base a hasSeenWelcome
                    NavigationStack {
                        if hasSeenWelcome {
                            ChatView()
                        } else {
                            WelcomeView(
                                hasSeenWelcome: $hasSeenWelcome,
                                searchQuery: $initialSearchQuery
                            )
                        }
                    }
                    .tabItem {
                        Image(systemName: "bubble.left.and.bubble.right")
                        Text("Chat")
                    }

                    NavigationStack { FavoritesTabView() }
                        .tabItem {
                            Image(systemName: "heart.fill")
                            Text("Preferiti")
                        }
                    
                    NavigationStack { TastingsTabView() }
                        .tabItem {
                            Image(systemName: "list.clipboard.fill")
                            Text("Degustazioni")
                        }
                }
                // ✅ Forza colori corretti di navigation bar e tab bar
                .toolbarColorScheme(.light, for: .navigationBar)
                .toolbarColorScheme(.light, for: .tabBar)
            }
        }
    }
}
