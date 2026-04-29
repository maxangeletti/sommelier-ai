//
//  FavoritesTabView.swift
//  SommelierAI
//

import SwiftUI

struct FavoritesTabView: View {
    @EnvironmentObject private var favoritesStore: FavoritesStore
    @EnvironmentObject private var tierStore: TierStore   // ✅ NEW (T1)

    var body: some View {
        VStack(spacing: 0) {

            // Header con sorting e delete
            HStack {
                Spacer()

                Menu {
                    Picker("Ordina per", selection: $favoritesStore.sort) {
                        ForEach(allowedSortModes, id: \.self) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }
                    .pickerStyle(.inline)
                } label: {
                    Image(systemName: "arrow.up.arrow.down")
                        .foregroundColor(AppColors.textSecondary)
                }

                Button(role: .destructive) {
                    favoritesStore.clear()
                } label: {
                    Image(systemName: "trash")
                        .foregroundColor(AppColors.accentWine)
                }
            }
            .padding(.horizontal, Spacing.screenEdges)
            .padding(.vertical, Spacing.md)
            .background(AppColors.backgroundSecondary)

            Divider()

            if favoritesStore.sortedFavorites.isEmpty {
                VStack(spacing: Spacing.lg) {
                    Spacer()
                    
                    Image(systemName: "heart.slash")
                        .font(.system(size: 60))
                        .foregroundColor(AppColors.textMuted)

                    Text("Nessun preferito")
                        .font(Typography.title3)
                        .foregroundColor(AppColors.textSecondary)
                    
                    Text("Aggiungi vini dalla ricerca\nper trovarli qui")
                        .font(Typography.callout)
                        .foregroundColor(AppColors.textMuted)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, Spacing.xxl)
                    
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(AppColors.backgroundPrimary)

            } else {
                ScrollView {
                    VStack(spacing: Spacing.md) {
                        ForEach(favoritesStore.sortedFavorites) { wine in
                            favoriteRow(wine)
                        }
                    }
                    .padding(Spacing.screenEdges)
                }
                .background(AppColors.backgroundPrimary)
            }
        }
        .navigationTitle("Preferiti")

        // ✅ gating anche al boot della tab (coerenza immediata)
        .onAppear {
            enforceTierSortIfNeeded()
        }

        // ✅ T1: gating passivo (Free → solo A2/Relevance) in Preferiti
        .onChange(of: favoritesStore.sort) { _, newValue in
            guard tierStore.tier == .free else { return }
            if newValue != .relevance {
                favoritesStore.sort = .relevance
            }
        }
    }

    private var allowedSortModes: [ChatDomain.SortMode] {
        if tierStore.tier == .free {
            return [.relevance]
        } else {
            return ChatDomain.SortMode.allCases
        }
    }

    private func enforceTierSortIfNeeded() {
        guard tierStore.tier == .free else { return }
        if favoritesStore.sort != .relevance {
            favoritesStore.sort = .relevance
        }
    }

    // MARK: - Row

    private func favoriteRow(_ wine: WineCard) -> some View {
        NavigationLink {
            WineDetailView(wine: wine, userQuery: "")
        } label: {
            VStack(alignment: .leading, spacing: Spacing.sm) {

                HStack {
                    Text(wine.name)
                        .font(Typography.title3)
                        .foregroundColor(AppColors.textPrimary)
                        .lineLimit(2)

                    Spacer()

                    Button {
                        favoritesStore.toggle(wine)
                    } label: {
                        Image(systemName: "heart.fill")
                            .foregroundColor(AppColors.accentWine)
                    }
                    .buttonStyle(.plain)
                }

                if let price = wine.price {
                    Text(String(format: "€%.2f", price))
                        .font(Typography.price)
                        .foregroundColor(AppColors.primaryWine)
                }

                Text(wine.reason)
                    .font(Typography.callout)
                    .foregroundColor(AppColors.textSecondary)
                    .lineLimit(2)

                if let tags = wine.tags, !tags.isEmpty {
                    Text(tags.filter { let t = $0.lowercased().replacingOccurrences(of: "_", with: " "); return !["red","white","rose","rosso","bianco","rosato","ruby red","ruby","orange","low","medium","high","fermo","secco","dolce","amabile","frizzante","spumante","garnet","straw","golden","pink","pale"].contains(t) }.map { $0.replacingOccurrences(of: "_", with: " ") }.joined(separator: " • "))
                        .font(Typography.caption)
                        .foregroundColor(AppColors.textMuted)
                }
            }
            .padding(Spacing.cardPadding)
            .background(AppColors.cardBackground)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(AppColors.borderLight, lineWidth: 0.5)
            )
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
    }
}

/*
 Nota rapida (non cambio ora)
 - Il gating era solo “onChange”: ora è anche “onAppear” per evitare che un sort non consentito resti attivo in Free.
 - Ho filtrato le opzioni del Picker in Free per coerenza UX (mostro solo Relevance).
 - Non ho toccato FavoritesStore/sortedFavorites: assumo che sortedFavorites dipenda già da favoritesStore.sort.
*/
