//
//  WelcomeView.swift
//  SommelierAI
//
//  Created on 8 Aprile 2026
//  Schermata di onboarding con suggerimenti query
//

import SwiftUI
import Combine

struct WelcomeView: View {
    @Binding var hasSeenWelcome: Bool
    @Binding var searchQuery: String
    @FocusState private var isSearchFocused: Bool
    
    // ✅ Pool completo di suggerimenti con emoji (allineati a backend v1.8.4)
    private let allSuggestions: [(emoji: String, text: String)] = [
        ("🍷", "Un sangiovese di buona qualità"),
        ("💎", "Un vino intenso e strutturato sopra i 20€"),
        ("🥂", "Uno spumante brut per aperitivo"),
        ("🍾", "Uno Champagne sopra i 30€"),
        ("🍋", "Un bianco con sentori agrumati sotto i 15€"),
        ("⭐", "Un vino con buon rapporto qualità prezzo"),
        ("👔", "Un Barolo o Brunello per una cena importante"),
        ("🥩", "Un rosso corposo per carne alla griglia"),
        ("🐟", "Un bianco minerale per pesce crudo"),
        ("☀️", "Un rosato fresco per l'estate"),
        ("🎉", "Un Prosecco per brindare"),
        ("🍰", "Un passito per dessert"),
        ("🌱", "Un vino biologico sotto i 20€"),
        ("🌟", "Un Nebbiolo giovane e tannico")
    ]
    
    // ✅ Suggerimenti mostrati (4 random)
    @State private var displayedSuggestions: [(emoji: String, text: String)] = []
    
    // ✅ Animazione Sommy (4 frame loop)
    @State private var currentFrame = 1
    private let frameCount = 4
    
    // Timer più lento per animazione fluida
    private let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    
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
            VStack(spacing: Spacing.lg) {
                // ✅ Avatar Sommy ANIMATO - centrato e stabile
                ZStack {
                    if let _ = UIImage(named: "sommy_frame_1") {
                        Image("sommy_frame_\(currentFrame)")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 200, height: 200)
                            .id(currentFrame) // Forza re-render smooth
                            .transition(.opacity)
                            .onReceive(timer) { _ in
                                currentFrame = (currentFrame % frameCount) + 1
                            }
                    } else {
                        // Fallback: usa sommy_avatar se i frame non esistono
                        Image("sommy_avatar")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 200, height: 200)
                    }
                }
                .frame(width: 200, height: 200)
                .clipped() // Evita overflow durante transizione
                
                Text("SommelierAI")
                    .font(Typography.largeTitle)
                    .foregroundColor(AppColors.textPrimary)
                
                Text("Il tuo sommelier personale digitale")
                    .font(Typography.callout)
                    .foregroundColor(AppColors.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, Spacing.xxl)
            }
            .padding(.top, 60)
            .padding(.bottom, 40)
            
            // Titolo suggerimenti
            Text("Prova a chiedere a Sommy:")
                .font(.system(size: 18, weight: .semibold))
                .foregroundColor(AppColors.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 24)
                .padding(.bottom, 16)
            
            // Suggerimenti come bottoni
            VStack(spacing: 12) {
                ForEach(displayedSuggestions, id: \.text) { suggestion in
                    SuggestionButton(
                        emoji: suggestion.emoji,
                        text: suggestion.text,
                        action: {
                            searchQuery = suggestion.text
                            proceedToChat()
                        }
                    )
                }
            }
            .padding(.horizontal, 24)
            .onAppear {
                // ✅ Scegli 4 suggerimenti random al primo load
                if displayedSuggestions.isEmpty {
                    displayedSuggestions = Array(allSuggestions.shuffled().prefix(4))
                }
            }
            
            Spacer()
            
            // Campo di ricerca in basso
            HStack(spacing: 12) {
                TextField("Scrivi che vino cerchi...", text: $searchQuery)
                    .font(.system(size: 16))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 24)
                            .fill(AppColors.cardBackground)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(AppColors.borderLight, lineWidth: 1)
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
                            .fill(AppColors.primaryWine)
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
        .background(AppColors.backgroundPrimary)
    }
    
    private func proceedToChat() {
        hasSeenWelcome = true
    }
}

// MARK: - Suggestion Button Component (con animazioni)

struct SuggestionButton: View {
    let emoji: String
    let text: String
    let action: () -> Void
    
    @State private var isPressed = false
    
    var body: some View {
        Button(action: {
            // Haptic feedback
            let impact = UIImpactFeedbackGenerator(style: .light)
            impact.impactOccurred()
            
            // Action
            action()
        }) {
            HStack(spacing: 12) {
                // Emoji icona grande
                Text(emoji)
                    .font(.system(size: 28))
                
                Text(text)
                    .font(.system(size: 16))
                    .foregroundColor(AppColors.textPrimary)
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .font(.system(size: 14))
                    .foregroundColor(AppColors.textSecondary)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(AppColors.cardBackground)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(AppColors.borderLight, lineWidth: 1)
            )
        }
        .buttonStyle(ScaleButtonStyle())
    }
}

// MARK: - Scale Button Style (Spring Animation)

struct ScaleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.6), value: configuration.isPressed)
    }
}

// Preview
#Preview {
    WelcomeView(
        hasSeenWelcome: .constant(false),
        searchQuery: .constant("")
    )
}
