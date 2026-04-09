//
//  APIClient.swift
//  SommelierAI
//
//  PURPOSE
//  Networking client: performs HTTP requests and consumes SSE streams.
//  Responsible for parsing/decoding transport payloads into canonical domain models.
//
//  SCOPE
//  - Request building (HTTP)
//  - SSE stream parsing (line/event handling)
//  - Decoding using existing models:
//    - APIStreamEvent (from APIStreamEvent.swift)
//    - SearchResponse/WineCard/SearchMeta (from Models.swift)
//
//  OUT OF SCOPE (MUST NOT BE DEFINED HERE)
//  - Domain model definitions (WineCard, SearchResponse, SearchMeta)
//  - Chat domain definitions (ChatDomain.Message)
//  - UI rendering logic (ChatView)
//
//  RULES
//  - Do not introduce “temporary” duplicate structs for decoding.
//  - If decoding fails, fix the envelope (APIStreamEvent) or the canonical model (Models.swift), not by shadowing types here.
//  - Keep this file focused on transport + decoding + delivery to ViewModels.
//
//

import Foundation

// MARK: - SSE Event Model (ALLINEATO AL BACKEND)
// ✅ PATCH 7c: modello SSE fuori da APIClient per evitare issue actor/Decodable

final class APIClient {
    private let baseURL = URL(string: "https://sommelier-ai.onrender.com")!
    private let timeout: TimeInterval = 120

    private struct SuggestionsResponse: Decodable {
        let suggestions: [String]
    }

    // MARK: - POST /search
    func search(query: String, sortMode: String = "relevance") async throws -> SearchResponse {
        var req = URLRequest(url: baseURL.appendingPathComponent("/search"))
        req.httpMethod = "POST"
        req.timeoutInterval = timeout
        req.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData  // ✅ DISABILITA CACHE
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = [
            "query": query,
            "sort": sortMode,
            "limit": 20
        ]

        #if DEBUG
        body["debug"] = true
        #endif

        req.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])
        print("🔎 REQUEST /search sort=\(sortMode) query=\(query) explain=true")

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }

        let decoded = try JSONDecoder().decode(SearchResponse.self, from: data)

        let head = (decoded.results.prefix(8)).map { w in
            let m = (w.match_score ?? w.__match_score) ?? -1
            return "\(w.name) m=\(String(format: "%.2f", m))"
        }
        print("⬇️ /search response head:", head.joined(separator: " | "))

        return decoded
    }

    func search(query: String, sort: ChatDomain.SortMode) async throws -> SearchResponse {
        try await search(query: query, sortMode: sort.apiValue)
    }

    // MARK: - SSE /search_stream (robust parser by \n\n)
    func searchStream(
        query: String,
        sortMode: String = "relevance",
        limit: Int = 20
    ) -> AsyncThrowingStream<APIStreamEvent, Error> {

        var items: [URLQueryItem] = [
            URLQueryItem(name: "query", value: query),
            URLQueryItem(name: "sort", value: sortMode),
            URLQueryItem(name: "limit", value: String(limit))
        ]

        #if DEBUG
        items.append(URLQueryItem(name: "debug", value: "true"))
        #endif

        let url = baseURL
            .appendingPathComponent("/search_stream")
            .appendingQueryItems(items)

        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = timeout
        req.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData  // ✅ DISABILITA CACHE
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        req.setValue("no-cache", forHTTPHeaderField: "Cache-Control")

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, resp) = try await URLSession.shared.bytes(for: req)

                    guard let http = resp as? HTTPURLResponse,
                          (200...299).contains(http.statusCode) else {
                        throw URLError(.badServerResponse)
                    }

                    var dataLines: [String] = []

                    func flushEvent() {
                        guard !dataLines.isEmpty else { return }

                        let payload = dataLines
                            .joined(separator: "\n")
                            .trimmingCharacters(in: .whitespacesAndNewlines)

                        dataLines.removeAll(keepingCapacity: true)

                        if payload == "[DONE]" {
                            continuation.finish()
                            return
                        }

                        guard let data = payload.data(using: .utf8) else { return }

                        Task { @MainActor in
                            do {
                                let ev = try JSONDecoder().decode(APIStreamEvent.self, from: data)

                                // 🔍 DEBUG BUILD CHECK
                                let t = ev.type.lowercased()
                                if t == "final" {
                                    print("[SSE FINAL] build=\(ev.meta?.build_id ?? "nil") results=\(ev.results?.count ?? ev.wines?.count ?? -1)")
                                } else if t == "delta", ev.wine != nil {
                                    print("[SSE DELTA] wine")
                                } else {
                                    print("[SSE \(ev.type)]")
                                }

                                continuation.yield(ev)

                            } catch {
                                print("[SSE decode error]", error)
                            }
                        }
                    }

                    var buffer = Data()
                    buffer.reserveCapacity(32_768)

                    let delimLF = Data([0x0A, 0x0A])                   // \n\n
                    let delimCRLF = Data([0x0D, 0x0A, 0x0D, 0x0A])     // \r\n\r\n

                    for try await b in bytes {
                        if Task.isCancelled {
                            continuation.finish()
                            return
                        }

                        buffer.append(b)

                        while true {
                            if let r = buffer.range(of: delimCRLF) {
                                let eventData = buffer.subdata(in: 0..<r.lowerBound)
                                buffer.removeSubrange(0..<r.upperBound)

                                if let eventStr = String(data: eventData, encoding: .utf8) {
                                    dataLines.removeAll(keepingCapacity: true)

                                    let lines = eventStr
                                        .replacingOccurrences(of: "\r\n", with: "\n")
                                        .split(separator: "\n", omittingEmptySubsequences: false)

                                    for lSub in lines {
                                        let l = String(lSub)
                                        if l.hasPrefix("data:") {
                                            var chunk = l.dropFirst(5)
                                            if chunk.first == " " { chunk = chunk.dropFirst() }
                                            dataLines.append(String(chunk))
                                        }
                                    }

                                    flushEvent()
                                }
                                continue
                            }

                            if let r = buffer.range(of: delimLF) {
                                let eventData = buffer.subdata(in: 0..<r.lowerBound)
                                buffer.removeSubrange(0..<r.upperBound)

                                if let eventStr = String(data: eventData, encoding: .utf8) {
                                    dataLines.removeAll(keepingCapacity: true)

                                    let lines = eventStr
                                        .replacingOccurrences(of: "\r\n", with: "\n")
                                        .split(separator: "\n", omittingEmptySubsequences: false)

                                    for lSub in lines {
                                        let l = String(lSub)
                                        if l.hasPrefix("data:") {
                                            var chunk = l.dropFirst(5)
                                            if chunk.first == " " { chunk = chunk.dropFirst() }
                                            dataLines.append(String(chunk))
                                        }
                                    }

                                    flushEvent()
                                }
                                continue
                            }

                            break
                        }
                    }

                    continuation.finish()

                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    func searchStream(
        query: String,
        sort: ChatDomain.SortMode,
        limit: Int = 20
    ) -> AsyncThrowingStream<APIStreamEvent, Error> {
        searchStream(query: query, sortMode: sort.apiValue, limit: limit)
    }

    // MARK: - Suggestions
    func suggestions() async throws -> [String] {
        let url = baseURL.appendingPathComponent("/suggestions")
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(SuggestionsResponse.self, from: data).suggestions
    }
    
    // MARK: - Tasting Notes (LLM Step 3)
    /// Chiama GET /wine/{id}/tasting_notes per ottenere note degustazione LLM
    func getTastingNotes(wineId: String, query: String = "") async throws -> String {
        var components = URLComponents(url: baseURL.appendingPathComponent("/wine/\(wineId)/tasting_notes"), resolvingAgainstBaseURL: false)!
        if !query.isEmpty {
            components.queryItems = [URLQueryItem(name: "query", value: query)]
        }
        
        guard let url = components.url else {
            throw URLError(.badURL)
        }
        
        let (data, resp) = try await URLSession.shared.data(from: url)
        guard let http = resp as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        
        struct TastingNotesResponse: Decodable {
            let wine_id: String
            let tasting_notes: String
        }
        
        let decoded = try JSONDecoder().decode(TastingNotesResponse.self, from: data)
        return decoded.tasting_notes
    }
    
    // MARK: - Similar Wines (Gli Imperdibili)
    /// Chiama GET /wine/{id}/similar per ottenere vini simili (stessa denomination)
    func getSimilarWines(wineId: String, limit: Int = 3) async throws -> [WineCard] {
        var components = URLComponents(url: baseURL.appendingPathComponent("/wine/\(wineId)/similar"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "limit", value: String(limit))]
        
        guard let url = components.url else {
            throw URLError(.badURL)
        }
        
        let (data, resp) = try await URLSession.shared.data(from: url)
        guard let http = resp as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        
        struct SimilarWinesResponse: Decodable {
            let similar_wines: [WineCard]
        }
        
        let decoded = try JSONDecoder().decode(SimilarWinesResponse.self, from: data)
        return decoded.similar_wines
    }
}

// MARK: - URL helper
private extension URL {
    func appendingQueryItems(_ items: [URLQueryItem]) -> URL {
        var c = URLComponents(url: self, resolvingAgainstBaseURL: false)!
        c.queryItems = (c.queryItems ?? []) + items
        return c.url!
    }
}
