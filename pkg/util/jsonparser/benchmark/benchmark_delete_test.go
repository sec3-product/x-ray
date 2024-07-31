package benchmark

import (
	"testing"

	"github.com/buger/jsonparser"
)

func BenchmarkDeleteSmall(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data1 := []byte(`{ "instanceId": 1, "ip": "10.10.10.10", "services": [ { "id": 1, "name": "srv1" } ] }`)
		_ = jsonparser.Delete(data1, "services")
	}
}

func BenchmarkDeleteNested(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data1 := []byte(`{ "instanceId": 1, "ip": "10.10.10.10", "services": [ { "id": 1, "name": "srv1" } ] }`)
		_ = jsonparser.Delete(data1, "services", "id")
	}
}

func BenchmarkDeleteLarge(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data1 := []byte(`{"adsEnabled":true,"assetGroup":{"id":"4131","logoURL":"https://www.gannett-cdn.com/sites/usatoday/images/blogs/talkingtech/logo_front_v2.png","name":"talkingtech","siteCode":"USAT","siteId":"1","siteName":"USA TODAY","sstsId":"c67ad92a-3c9b-4817-9030-9357a9c2a86e","type":"blog","URL":"/tech/talkingtech"},"authoringBehavior":"text","authoringTypeCode":"blog","awsPath":"tech/talkingtech","backfillDate":"2018-10-30T14:56:31.522Z","byline":"Mike Snider","contentProtectionState":"free","contentSourceCode":"USAT","contributors":[{"id":"1071","name":"Mike Snider"}],"createDate":"2018-10-30T13:58:41.194Z","createSystem":"Presto Next","createUser":"msnider","eventDate":"2018-10-30T15:09:50.43Z","excludeFromMobile":false,"fronts":[{"id":"206","name":"tech","recommendedDate":"2018-10-30T15:09:50.399Z","type":"section-front"},{"id":"1012186","name":"tech_talkingtech","recommendedDate":"2018-10-30T15:09:50.399Z","type":"section-front"},{"id":"196","name":"money","recommendedDate":"2018-10-30T15:09:50.399Z","type":"section-front"},{"id":"156","name":"home","recommendedDate":"2018-10-30T15:09:50.399Z","type":"section-front"},{"id":"156","name":"home","recommendedDate":"2018-10-30T15:09:50.399Z","type":"section-front"}],"geoTag":{"attributes":{"lat":"","long":""},"id":""},"headline":"'Red Dead Redemption 2' rakes in $725M for Rockstar Games in blockbuster weekend debut","id":"1817435002","initialPublishDate":"2018-10-30T14:56:31.522Z","isEvergreen":false,"links":{"assets":[{"id":"1763879002","overrides":{},"position":1,"relationshipTypeFlags":"PromoImage"},{"id":"1764652002","overrides":{},"position":2,"relationshipTypeFlags":"Undefined"},{"id":"1765924002","overrides":{},"position":3,"relationshipTypeFlags":"Undefined"}],"photoId":"1763879002"},"pageURL":{"long":"http://www.usatoday.com/story/tech/talkingtech/2018/10/30/red-dead-redemption-2-makes-725-million-debut-rockstar-games/1817435002/","short":"http://www.usatoday.com/story/tech/talkingtech/2018/10/30/red-dead-redemption-2-makes-725-million-debut-rockstar-games/1817435002/"},"promoBrief":"Video game \"Red Dead Redemption 2\" corralled blockbuster sales of $725 million in its first three days, according to publisher Rockstar Games.","propertyDisplayName":"USA TODAY","propertyId":"1","propertyName":"USATODAY","publication":"USA TODAY","publishDate":"2018-10-30T15:09:50.399Z","publishSystem":"authoring","publishUser":"geronimo-publish-handler","readerCommentsEnabled":false,"schemaVersion":"0.11.20","shortHeadline":"'Red Dead Redemption 2' corrals $725M in sales","source":"USA TODAY","ssts":{"leafName":"talkingtech","path":"tech/talkingtech","section":"tech","subsection":"talkingtech","taxonomyEntityDisplayName":"Talking Tech","topic":"","subtopic":""},"statusName":"published","tags":[{"id":"855b0686-b2d8-4d98-b5f4-fcacf713047b","isPrimary":true,"name":"Talking Tech","path":"USAT TOPICS/USAT Science and technology/Talking Tech","taggingStatus":"UserTagged","vocabulary":"Topics"},{"id":"5dd5b5f2-9594-4aae-83c8-1ebb8aa50767","name":"Rockstar Games","path":"Candidates/Rockstar Games","taggingStatus":"UserTagged","vocabulary":"Companies"},{"id":"ceff0ffa-451d-46ae-8c4f-f958264b165e","name":"Video Games","path":"Consumer Products/Video Games","taggingStatus":"UserTagged","vocabulary":"Subjects"},{"id":"d59ddfbc-2afe-40e3-a9a2-5debe530dc5f","name":"Redemption","path":"Religious Organizations/Redemption","taggingStatus":"AutoTagged","vocabulary":"Organizations"},{"id":"09f4e1a7-50e7-4fc5-b318-d300acc4718f","name":"Success","path":"Emotions/Success","taggingStatus":"AutoTagged","vocabulary":"SubjectCodes"},{"id":"7095bb07-b172-434b-a4eb-8856263ad949","name":"Overall Positive","path":"Emotions/Overall Positive","taggingStatus":"AutoTagged","vocabulary":"SubjectCodes"},{"id":"d2cb2465-3a24-4104-8569-31785b515f62","name":"Sony","path":"Corporations/Sony","taggingStatus":"AutoTagged","vocabulary":"Companies"},{"id":"9b993d1c-2a6d-4279-acb3-ecac95d77320","name":"Amusement","path":"Emotions/Amusement","taggingStatus":"AutoTagged","vocabulary":"SubjectCodes"}],"title":"Red Dead Redemption 2 makes $725 million in debut for Rockstar Games","updateDate":"2018-10-30T15:09:50.43Z","updateUser":"mhayes","aggregateId":"acba765c-c573-42af-929f-26ea5920b932","body":{"desktop":[{"type":"asset","value":"1763879002"},{"type":"text","value":"<p>Rockstar Games has another hard-boiled hit on its hands.</p>"},{"type":"text","value":"<p>Old West adventure game &quot;Red Dead Redemption 2,&quot; which landed Friday, lassoed $725 million in sales worldwide in its first three days.</p>"},{"type":"text","value":"<p>That places the massive explorable open-world game&nbsp;as the&nbsp;No. 2 game out of the gate, just behind Rockstar&#39;s &quot;Grand Theft Auto V,&quot; the biggest seller of all time, which took in $1 billion in its first three days when it launched  on Sept. 17, 2013.</p>"},{"type":"text","value":"<p>&quot;GTA V&quot; has gone on to make more money than any other single game title, selling nearly 100 million copies and reaping $6 billion in revenue, according to <a href=\"https://www.marketwatch.com/story/this-violent-videogame-has-made-more-money-than-any-movie-ever-2018-04-06\">MarketWatch</a>.</p>"},{"type":"text","value":"<p>The&nbsp;three-day start makes &quot;Red Dead Redemption 2&quot; the single-biggest opening weekend in &quot;the history of entertainment,&quot; Rockstar said&nbsp;in a press release detailing the game&#39;s achievements. That means the three-day sales for the game, prices of which start at $59.99 (rated Mature for those 17-up), surpasses opening weekends for blockbuster movies such as &quot;Avengers: Infinity War&quot; and &quot;Star Wars: The Force Awakens.&quot;</p>"},{"type":"text","value":"<p><span class=\"exclude-from-newsgate\"><strong style=\"margin-right:3px;\">More: </strong><a href=\"http://www.usatoday.com/story/tech/gaming/2018/10/26/red-dead-redemption-2-rockstar-games-western-classic/1650295002/\">&#39;Red Dead Redemption 2&#39;: First impressions from once upon a time in the West</a></span><br/>\n<br/>\n<span class=\"exclude-from-newsgate\"><strong style=\"margin-right:3px;\">More: </strong><a href=\"http://www.usatoday.com/story/tech/talkingtech/2018/10/30/sony-playstation-classic-games-list/1816900002/\">Sony lists the 20 games coming to PlayStation Classic retro video game console</a></span></p>"},{"type":"text","value":"<p>&quot;Red Dead Redemption 2&quot; also tallied the biggest full game sales marks for one and for three days on Sony&#39;s PlayStation Network, Rockstar said. It was also the most preordered game on Sony&#39;s online network.</p>"},{"type":"text","value":"<p>Reviews for the game have rated it&nbsp;among the best&nbsp;ever. It&nbsp;earned a 97 on Metacritic, earning it a tie for No. 6 all-time, along with games such as &quot;Super Mario Galaxy&quot; and &quot;GTA V.&quot;&nbsp;&nbsp;</p>"},{"type":"text","value":"<p>Piper Jaffray &amp; Co.&nbsp;senior research analyst&nbsp;Michael Olson estimated Rockstar sold about 11 million copies in its first three days. That means Olson&#39;s initial&nbsp;estimate of Rockstar selling 15.5 million copies of &quot;Red Dead 2&quot; in its fiscal year, which ends in March 2019, &quot;appears conservative,&quot; he said in a note to investors Tuesday.</p>"},{"type":"text","value":"<p>&quot;Clearly, with RDR2 first weekend sell-through exceeding CoD: Black Ops 4, it now appears RDR2 estimates may have been overly conservative,&quot; Olson wrote.</p>"},{"type":"text","value":"<p>Shares of Rockstar&rsquo;s parent company Take-Two Interactive (<a href=\"https://www.usatoday.com/money/lookup/stocks/TTWO/\">TTWO</a>) rose about 8 percent in early trading Tuesday to $120.55.</p>"},{"type":"asset","value":"1765924002"},{"type":"text","value":"<p><em>Follow USA TODAY reporter Mike Snider on Twitter: <a href=\"http://twitter.com/MikeSnider\">@MikeSnider</a>.</em></p>"}],"mobile":[{"type":"asset","value":"1763879002"},{"type":"text","value":"<p>Rockstar Games has another hard-boiled hit on its hands.</p>"},{"type":"text","value":"<p>Old West adventure game &quot;Red Dead Redemption 2,&quot; which landed Friday, lassoed $725 million in sales worldwide in its first three days.</p>"},{"type":"text","value":"<p>That places the massive explorable open-world game&nbsp;as the&nbsp;No. 2 game out of the gate, just behind Rockstar&#39;s &quot;Grand Theft Auto V,&quot; the biggest seller of all time, which took in $1 billion in its first three days when it launched  on Sept. 17, 2013.</p>"},{"type":"text","value":"<p>&quot;GTA V&quot; has gone on to make more money than any other single game title, selling nearly 100 million copies and reaping $6 billion in revenue, according to <a href=\"https://www.marketwatch.com/story/this-violent-videogame-has-made-more-money-than-any-movie-ever-2018-04-06\">MarketWatch</a>.</p>"},{"type":"text","value":"<p>The&nbsp;three-day start makes &quot;Red Dead Redemption 2&quot; the single-biggest opening weekend in &quot;the history of entertainment,&quot; Rockstar said&nbsp;in a press release detailing the game&#39;s achievements. That means the three-day sales for the game, prices of which start at $59.99 (rated Mature for those 17-up), surpasses opening weekends for blockbuster movies such as &quot;Avengers: Infinity War&quot; and &quot;Star Wars: The Force Awakens.&quot;</p>"},{"type":"text","value":"<p><span class=\"exclude-from-newsgate\"><strong style=\"margin-right:3px;\">More: </strong><a href=\"http://www.usatoday.com/story/tech/gaming/2018/10/26/red-dead-redemption-2-rockstar-games-western-classic/1650295002/\">&#39;Red Dead Redemption 2&#39;: First impressions from once upon a time in the West</a></span><br/>\n<br/>\n<span class=\"exclude-from-newsgate\"><strong style=\"margin-right:3px;\">More: </strong><a href=\"http://www.usatoday.com/story/tech/talkingtech/2018/10/30/sony-playstation-classic-games-list/1816900002/\">Sony lists the 20 games coming to PlayStation Classic retro video game console</a></span></p>"},{"type":"text","value":"<p>&quot;Red Dead Redemption 2&quot; also tallied the biggest full game sales marks for one and for three days on Sony&#39;s PlayStation Network, Rockstar said. It was also the most preordered game on Sony&#39;s online network.</p>"},{"type":"text","value":"<p>Reviews for the game have rated it&nbsp;among the best&nbsp;ever. It&nbsp;earned a 97 on Metacritic, earning it a tie for No. 6 all-time, along with games such as &quot;Super Mario Galaxy&quot; and &quot;GTA V.&quot;&nbsp;&nbsp;</p>"},{"type":"text","value":"<p>Piper Jaffray &amp; Co.&nbsp;senior research analyst&nbsp;Michael Olson estimated Rockstar sold about 11 million copies in its first three days. That means Olson&#39;s initial&nbsp;estimate of Rockstar selling 15.5 million copies of &quot;Red Dead 2&quot; in its fiscal year, which ends in March 2019, &quot;appears conservative,&quot; he said in a note to investors Tuesday.</p>"},{"type":"text","value":"<p>&quot;Clearly, with RDR2 first weekend sell-through exceeding CoD: Black Ops 4, it now appears RDR2 estimates may have been overly conservative,&quot; Olson wrote.</p>"},{"type":"text","value":"<p>Shares of Rockstar&rsquo;s parent company Take-Two Interactive (<a href=\"https://www.usatoday.com/money/lookup/stocks/TTWO/\">TTWO</a>) rose about 8 percent in early trading Tuesday to $120.55.</p>"},{"type":"asset","value":"1765924002"},{"type":"text","value":"<p><em>Follow USA TODAY reporter Mike Snider on Twitter: <a href=\"http://twitter.com/MikeSnider\">@MikeSnider</a>.</em></p>"}]},"fullText":"<p><img assetid=\"1763879002\" assettype=\"image\"/></p>\n\n<p>Rockstar Games has another hard-boiled hit on its hands.</p>\n\n<p>Old West adventure game &quot;Red Dead Redemption 2,&quot; which landed Friday, lassoed $725 million in sales worldwide in its first three days.</p>\n\n<p>That places the massive explorable open-world game&nbsp;as the&nbsp;No. 2 game out of the gate, just behind Rockstar&#39;s &quot;Grand Theft Auto V,&quot; the biggest seller of all time, which took in $1 billion in its first three days when it launched  on Sept. 17, 2013.</p>\n\n<p>&quot;GTA V&quot; has gone on to make more money than any other single game title, selling nearly 100 million copies and reaping $6 billion in revenue, according to <a href=\"https://www.marketwatch.com/story/this-violent-videogame-has-made-more-money-than-any-movie-ever-2018-04-06\">MarketWatch</a>.</p>\n\n<p>The&nbsp;three-day start makes &quot;Red Dead Redemption 2&quot; the single-biggest opening weekend in &quot;the history of entertainment,&quot; Rockstar said&nbsp;in a press release detailing the game&#39;s achievements. That means the three-day sales for the game, prices of which start at $59.99 (rated Mature for those 17-up), surpasses opening weekends for blockbuster movies such as &quot;Avengers: Infinity War&quot; and &quot;Star Wars: The Force Awakens.&quot;</p>\n\n<p><span class=\"exclude-from-newsgate\"><strong style=\"margin-right:3px;\">More: </strong><a href=\"http://www.usatoday.com/story/tech/gaming/2018/10/26/red-dead-redemption-2-rockstar-games-western-classic/1650295002/\">&#39;Red Dead Redemption 2&#39;: First impressions from once upon a time in the West</a></span><br/>\n<br/>\n<span class=\"exclude-from-newsgate\"><strong style=\"margin-right:3px;\">More: </strong><a href=\"http://www.usatoday.com/story/tech/talkingtech/2018/10/30/sony-playstation-classic-games-list/1816900002/\">Sony lists the 20 games coming to PlayStation Classic retro video game console</a></span></p>\n\n<p>&quot;Red Dead Redemption 2&quot; also tallied the biggest full game sales marks for one and for three days on Sony&#39;s PlayStation Network, Rockstar said. It was also the most preordered game on Sony&#39;s online network.</p>\n\n<p>Reviews for the game have rated it&nbsp;among the best&nbsp;ever. It&nbsp;earned a 97 on Metacritic, earning it a tie for No. 6 all-time, along with games such as &quot;Super Mario Galaxy&quot; and &quot;GTA V.&quot;&nbsp;&nbsp;</p>\n\n<p>Piper Jaffray &amp; Co.&nbsp;senior research analyst&nbsp;Michael Olson estimated Rockstar sold about 11 million copies in its first three days. That means Olson&#39;s initial&nbsp;estimate of Rockstar selling 15.5 million copies of &quot;Red Dead 2&quot; in its fiscal year, which ends in March 2019, &quot;appears conservative,&quot; he said in a note to investors Tuesday.</p>\n\n<p>&quot;Clearly, with RDR2 first weekend sell-through exceeding CoD: Black Ops 4, it now appears RDR2 estimates may have been overly conservative,&quot; Olson wrote.</p>\n\n<p>Shares of Rockstar&rsquo;s parent company Take-Two Interactive (<a href=\"https://www.usatoday.com/money/lookup/stocks/TTWO/\">TTWO</a>) rose about 8 percent in early trading Tuesday to $120.55.</p>\n\n<p><img assetid=\"1765924002\" assettype=\"embed\"/></p>\n\n<p><em>Follow USA TODAY reporter Mike Snider on Twitter: <a href=\"http://twitter.com/MikeSnider\">@MikeSnider</a>.</em></p>\n","layoutPriorityAssetId":"1763879002","seoTitle":"Red Dead Redemption 2 makes $725 million in debut for Rockstar Games","type":"text","versionHash":"fe60306b2e7574a8d65e690753deb666"}`)
		_ = jsonparser.Delete(data1, "body")
	}
}