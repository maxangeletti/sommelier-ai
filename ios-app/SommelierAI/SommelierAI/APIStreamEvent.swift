//
//  APIStreamEvent.swift
//  SommelierAI
//
//  PURPOSE
//  SSE (Server-Sent Events) transport layer models.
//  Defines the decodable event envelope(s) received over the streaming connection.
//
//  SCOPE (OWNED TYPES)
//  - APIStreamEvent (and stream-only envelopes like delta/final/error, if present)
//
//  OUT OF SCOPE (MUST NOT BE DEFINED HERE)
//  - Domain models (WineCard, SearchResponse, SearchMeta) -> defined in Models.swift
//    (They can be referenced as payload types, but never duplicated here.)
//  - Chat domain (ChatDomain.Message) -> defined in ChatTypes.swift
//  - Networking/parsing implementation -> defined in APIClient.swift
//
//  RULES
//  - Keep Decodable logic and stream envelope shape confined to this file.
//  - Any server schema change for SSE should usually only require edits here (and possibly APIClient parsing).
//


import Foundation

// MARK: - SSE Event Model (ALLINEATO AL BACKEND)
// ✅ PATCH 7c: modello SSE fuori da APIClient e fuori da contesti actor-isolated
struct APIStreamEvent: Decodable, Sendable {
    let type: String
    let text: String?
    let message: String?
    let wine: WineCard?          // delta
    let results: [WineCard]?     // final
    let wines: [WineCard]?       // compat legacy
    let meta: SearchMeta?        // debug (build_id)

    private enum CodingKeys: String, CodingKey {
        case type, text, message, wine, results, wines, meta
    }
}
