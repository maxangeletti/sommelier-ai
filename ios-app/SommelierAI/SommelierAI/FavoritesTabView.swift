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

            // Header
            HStack {
                Text("Preferiti")
                    .font(.title2)
                    .fontWeight(.semibold)

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
                }

                Button(role: .destructive) {
                    favoritesStore.clear()
                } label: {
                    Image(systemName: "trash")
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            Divider()

            if favoritesStore.sortedFavorites.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "heart")
                        .font(.system(size: 40))
                        .foregroundStyle(.secondary)

                    Text("Nessun preferito")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            } else {
                ScrollView {
                    VStack(spacing: 12) {
                        ForEach(favoritesStore.sortedFavorites) { wine in
                            favoriteRow(wine)
                        }
                    }
                    .padding(12)
                }
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
        VStack(alignment: .leading, spacing: 8) {

            HStack {
                Text(wine.name)
                    .font(.headline)

                Spacer()

                Button {
                    favoritesStore.toggle(wine)
                } label: {
                    Image(systemName: "heart.fill")
                        .foregroundStyle(.red)
                }
            }

            if let price = wine.price {
                Text(String(format: "€%.2f", price))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Text(wine.reason)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .lineLimit(3)

            if let tags = wine.tags, !tags.isEmpty {
                Text(tags.filter { !["red","white","rose","rosso","bianco","rosato","ruby_red","ruby red","low","medium","high","fermo","secco","dolce","amabile","frizzante","spumante"].contains($0.lowercased()) }.map { $0.replacingOccurrences(of: "_", with: " ") }.joined(separator: " • "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(Color.gray.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}

/*
 Nota rapida (non cambio ora)
 - Il gating era solo “onChange”: ora è anche “onAppear” per evitare che un sort non consentito resti attivo in Free.
 - Ho filtrato le opzioni del Picker in Free per coerenza UX (mostro solo Relevance).
 - Non ho toccato FavoritesStore/sortedFavorites: assumo che sortedFavorites dipenda già da favoritesStore.sort.
*/
