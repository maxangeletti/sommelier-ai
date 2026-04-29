//
//  EmptyStateView.swift
//  SommelierAI
//

import SwiftUI

enum EmptyStateType {
    case noResults
    case searchError
    case welcome
    
    var icon: String {
        switch self {
        case .noResults: return "magnifyingglass"
        case .searchError: return "exclamationmark.triangle"
        case .welcome: return "sparkles"
        }
    }
    
    var title: String {
        switch self {
        case .noResults: return "Nessun vino trovato"
        case .searchError: return "Qualcosa è andato storto"
        case .welcome: return "Inizia a cercare"
        }
    }
    
    func subtitle(didYouMean: [String] = []) -> String {
        switch self {
        case .noResults:
            return didYouMean.isEmpty 
                ? "Prova a riformulare la ricerca o usa termini più generici."
                : "Prova con una di queste ricerche simili:"
        case .searchError:
            return "Riprova tra qualche istante."
        case .welcome:
            return "Chiedi a Sommy di aiutarti a trovare il vino perfetto."
        }
    }
    
    var accentColor: Color {
        switch self {
        case .noResults: return AppColors.textSecondary
        case .searchError: return .orange
        case .welcome: return AppColors.primaryWine
        }
    }
}

struct EmptyStateView: View {
    let type: EmptyStateType
    let didYouMeanSuggestions: [String]
    let onSuggestionTap: ((String) -> Void)?
    
    init(
        type: EmptyStateType,
        didYouMeanSuggestions: [String] = [],
        onSuggestionTap: ((String) -> Void)? = nil
    ) {
        self.type = type
        self.didYouMeanSuggestions = didYouMeanSuggestions
        self.onSuggestionTap = onSuggestionTap
    }
    
    var body: some View {
        VStack(spacing: 24) {
            ZStack {
                Circle()
                    .fill(type.accentColor.opacity(0.1))
                    .frame(width: 120, height: 120)
                
                Image(systemName: type.icon)
                    .font(.system(size: 44, weight: .medium))
                    .foregroundColor(type.accentColor)
            }
            
            VStack(spacing: 8) {
                Text(type.title)
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundColor(AppColors.textPrimary)
                
                Text(type.subtitle(didYouMean: didYouMeanSuggestions))
                    .font(.system(size: 16))
                    .foregroundColor(AppColors.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }
            
            if !didYouMeanSuggestions.isEmpty {
                VStack(spacing: 12) {
                    ForEach(didYouMeanSuggestions.prefix(3), id: \.self) { suggestion in
                        Button(action: {
                            onSuggestionTap?(suggestion)
                        }) {
                            HStack {
                                Image(systemName: "sparkles")
                                    .font(.system(size: 14))
                                Text(suggestion)
                                    .font(.system(size: 15))
                                Spacer()
                                Image(systemName: "arrow.right")
                                    .font(.system(size: 12))
                            }
                            .foregroundColor(AppColors.textPrimary)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
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
                .padding(.horizontal, 24)
                .padding(.top, 8)
            }
        }
        .padding(.vertical, 60)
        .transition(.opacity.combined(with: .scale(scale: 0.95)))
    }
}

#Preview {
    EmptyStateView(
        type: .noResults,
        didYouMeanSuggestions: ["Chianti Classico", "Sangiovese Toscana"],
        onSuggestionTap: { _ in }
    )
    .background(AppColors.backgroundPrimary)
}
