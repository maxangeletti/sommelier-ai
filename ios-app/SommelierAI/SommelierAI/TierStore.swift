//
//  TierStore.swift
//  SommelierAI
//
//  Created by Massimiliano Angeletti on 22/02/26.
//

import Foundation
import Combine

enum UserTier: String, Codable {
    case free
    case premium
    case premiumPlus
}

@MainActor
final class TierStore: ObservableObject {

    @Published var tier: UserTier = .free {
        didSet { save() }
    }

    private let storageKey = "sommelier_user_tier_v1"

    init() {
        load()
//        tier = .premium   // DEBUG TEMPORANEO — rimuovere dopo i test
    }

    private func save() {
        UserDefaults.standard.set(tier.rawValue, forKey: storageKey)
    }

    private func load() {
        guard let raw = UserDefaults.standard.string(forKey: storageKey),
              let t = UserTier(rawValue: raw) else { return }
        tier = t
    }
}
