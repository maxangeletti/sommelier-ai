//
//  WelcomeView.swift
//  SommelierAI
//
//  Created on 8 Aprile 2026
//  Schermata di onboarding con suggerimenti query
//

import SwiftUI

struct WelcomeView: View {
    @Binding var hasSeenWelcome: Bool
    @Binding var searchQuery: String
    @FocusState private var isSearchFocused: Bool
    
    // Suggerimenti predefiniti
    private let suggestions = [
        "Vino per cena importante",
        "Rosso strutturato",
        "Vino sotto 20 euro",
        "Bianco fresco"
    ]
    
    private func proceedToChat() {
        // ✅ Salva query pendente per ChatView
        if !searchQuery.isEmpty {
            UserDefaults.standard.set(searchQuery, forKey: "pendingSearchQuery")
        }
        hasSeenWelcome = true
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header con logo e titolo
            VStack(spacing: 16) {
                // Logo placeholder (cerchio grigio con icona)
                ZStack {
                    Circle()
                        .fill(Color.gray.opacity(0.2))
                        .frame(width: 80, height: 80)
                    
                    Image(systemName: "wineglass")
                        .font(.system(size: 36))
                        .foregroundColor(.gray)
                }
                
                Text("SommelierAI")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.primary)
            }
            .padding(.top, 60)
            .padding(.bottom, 40)
            
            // Titolo suggerimenti
            Text("Prova a chiedere a Sommy:")
                .font(.system(size: 18, weight: .semibold))
                .foregroundColor(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 24)
                .padding(.bottom, 16)
            
            // Suggerimenti come bottoni
            VStack(spacing: 12) {
                ForEach(suggestions, id: \.self) { suggestion in
                    Button(action: {
                        searchQuery = suggestion
                        proceedToChat()
                    }) {
                        HStack {
                            Text(suggestion)
                                .font(.system(size: 16))
                                .foregroundColor(.primary)
                            
                            Spacer()
                            
                            Image(systemName: "chevron.right")
                                .font(.system(size: 14))
                                .foregroundColor(.gray)
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 16)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.gray.opacity(0.08))
                        )
                    }
                }
            }
            .padding(.horizontal, 24)
            
            Spacer()
            
            // Campo di ricerca in basso
            HStack(spacing: 12) {
                TextField("Scrivi che vino cerchi...", text: $searchQuery)
                    .font(.system(size: 16))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 24)
                            .fill(Color.gray.opacity(0.1))
                    )
                    .focused($isSearchFocused)
                    .submitLabel(.search)
                    .onSubmit {
                        if !searchQuery.isEmpty {
                            proceedToChat()
                        }
                    }
                
                // Bottone cerca
                Button(action: {
                    if !searchQuery.isEmpty {
                        proceedToChat()
                    }
                }) {
                    ZStack {
                        Circle()
                            .fill(Color(red: 0.5, green: 0.2, blue: 0.3))
                            .frame(width: 48, height: 48)
                        
                        Image(systemName: "arrow.right")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.white)
                    }
                }
                .disabled(searchQuery.isEmpty)
                .opacity(searchQuery.isEmpty ? 0.5 : 1.0)
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 32)
        }
        .background(Color(UIColor.systemBackground))
    }
}

// Preview
#Preview {
    WelcomeView(
        hasSeenWelcome: .constant(false),
        searchQuery: .constant("")
    )
}
