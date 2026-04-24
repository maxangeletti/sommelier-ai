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
    
    // ✅ Mostra splash screen iniziale
    @State private var showingSplash = true
    
    init() {
        // Production: no debug reset
    }
    
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
            } else if hasSeenWelcome {
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
    }
}
