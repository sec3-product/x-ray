// Generated from Smalltalk.g4 by ANTLR 4.9
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class SmalltalkLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.9", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		SEPARATOR=1, STRING=2, COMMENT=3, BLOCK_START=4, BLOCK_END=5, CLOSE_PAREN=6, 
		OPEN_PAREN=7, PIPE2=8, PIPE=9, PERIOD=10, SEMI_COLON=11, BINARY_SELECTOR=12, 
		LT=13, GT=14, MINUS=15, RESERVED_WORD=16, IDENTIFIER=17, CARROT=18, COLON=19, 
		ASSIGNMENT=20, HASH=21, DOLLAR=22, EXP=23, HEX=24, LITARR_START=25, DYNDICT_START=26, 
		DYNARR_END=27, DYNARR_START=28, DIGIT=29, HEXDIGIT=30, KEYWORD=31, BLOCK_PARAM=32, 
		CHARACTER_CONSTANT=33;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"SEPARATOR", "STRING", "COMMENT", "BLOCK_START", "BLOCK_END", "CLOSE_PAREN", 
			"OPEN_PAREN", "PIPE2", "PIPE", "PERIOD", "SEMI_COLON", "BINARY_SELECTOR", 
			"LT", "GT", "MINUS", "RESERVED_WORD", "IDENTIFIER", "CARROT", "COLON", 
			"ASSIGNMENT", "HASH", "DOLLAR", "EXP", "HEX", "LITARR_START", "DYNDICT_START", 
			"DYNARR_END", "DYNARR_START", "DIGIT", "HEXDIGIT", "KEYWORD", "BLOCK_PARAM", 
			"CHARACTER_CONSTANT"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, null, null, null, "'['", "']'", "')'", "'('", "'||'", "'|'", "'.'", 
			"';'", null, "'<'", "'>'", "'-'", null, null, "'^'", "':'", "' :='", "'#'", 
			"'$'", "'e'", "'16r'", "'#('", "'#{'", "'}'", "'{'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "SEPARATOR", "STRING", "COMMENT", "BLOCK_START", "BLOCK_END", "CLOSE_PAREN", 
			"OPEN_PAREN", "PIPE2", "PIPE", "PERIOD", "SEMI_COLON", "BINARY_SELECTOR", 
			"LT", "GT", "MINUS", "RESERVED_WORD", "IDENTIFIER", "CARROT", "COLON", 
			"ASSIGNMENT", "HASH", "DOLLAR", "EXP", "HEX", "LITARR_START", "DYNDICT_START", 
			"DYNARR_END", "DYNARR_START", "DIGIT", "HEXDIGIT", "KEYWORD", "BLOCK_PARAM", 
			"CHARACTER_CONSTANT"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public SmalltalkLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "Smalltalk.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2#\u00c3\b\1\4\2\t"+
		"\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t \4!"+
		"\t!\4\"\t\"\3\2\3\2\3\3\3\3\7\3J\n\3\f\3\16\3M\13\3\3\3\3\3\3\4\3\4\7"+
		"\4S\n\4\f\4\16\4V\13\4\3\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t\3\t"+
		"\3\t\3\n\3\n\3\13\3\13\3\f\3\f\3\r\3\r\3\r\6\rn\n\r\r\r\16\ro\3\16\3\16"+
		"\3\17\3\17\3\20\3\20\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21"+
		"\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\5\21\u008d\n\21"+
		"\3\22\6\22\u0090\n\22\r\22\16\22\u0091\3\22\7\22\u0095\n\22\f\22\16\22"+
		"\u0098\13\22\3\23\3\23\3\24\3\24\3\25\3\25\3\25\3\26\3\26\3\27\3\27\3"+
		"\30\3\30\3\31\3\31\3\31\3\31\3\32\3\32\3\32\3\33\3\33\3\33\3\34\3\34\3"+
		"\35\3\35\3\36\3\36\3\37\3\37\3 \3 \3 \3!\3!\3!\3\"\3\"\3\"\5\"\u00c2\n"+
		"\"\4KT\2#\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27\r\31\16\33"+
		"\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-\30/\31\61\32\63\33\65\34\67"+
		"\359\36;\37= ?!A\"C#\3\2\t\5\2\13\f\17\17\"\"\t\2\'\',.\61\61>@BB^^\u0080"+
		"\u0080\5\2((//AA\4\2C\\c|\6\2\62;C\\aac|\3\2\62;\5\2\62;CHch\2\u00ce\2"+
		"\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2"+
		"\2\2\17\3\2\2\2\2\21\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2"+
		"\31\3\2\2\2\2\33\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2"+
		"\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2"+
		"\2\61\3\2\2\2\2\63\3\2\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2"+
		"\2\2\2=\3\2\2\2\2?\3\2\2\2\2A\3\2\2\2\2C\3\2\2\2\3E\3\2\2\2\5G\3\2\2\2"+
		"\7P\3\2\2\2\tY\3\2\2\2\13[\3\2\2\2\r]\3\2\2\2\17_\3\2\2\2\21a\3\2\2\2"+
		"\23d\3\2\2\2\25f\3\2\2\2\27h\3\2\2\2\31m\3\2\2\2\33q\3\2\2\2\35s\3\2\2"+
		"\2\37u\3\2\2\2!\u008c\3\2\2\2#\u008f\3\2\2\2%\u0099\3\2\2\2\'\u009b\3"+
		"\2\2\2)\u009d\3\2\2\2+\u00a0\3\2\2\2-\u00a2\3\2\2\2/\u00a4\3\2\2\2\61"+
		"\u00a6\3\2\2\2\63\u00aa\3\2\2\2\65\u00ad\3\2\2\2\67\u00b0\3\2\2\29\u00b2"+
		"\3\2\2\2;\u00b4\3\2\2\2=\u00b6\3\2\2\2?\u00b8\3\2\2\2A\u00bb\3\2\2\2C"+
		"\u00be\3\2\2\2EF\t\2\2\2F\4\3\2\2\2GK\7)\2\2HJ\13\2\2\2IH\3\2\2\2JM\3"+
		"\2\2\2KL\3\2\2\2KI\3\2\2\2LN\3\2\2\2MK\3\2\2\2NO\7)\2\2O\6\3\2\2\2PT\7"+
		"$\2\2QS\13\2\2\2RQ\3\2\2\2SV\3\2\2\2TU\3\2\2\2TR\3\2\2\2UW\3\2\2\2VT\3"+
		"\2\2\2WX\7$\2\2X\b\3\2\2\2YZ\7]\2\2Z\n\3\2\2\2[\\\7_\2\2\\\f\3\2\2\2]"+
		"^\7+\2\2^\16\3\2\2\2_`\7*\2\2`\20\3\2\2\2ab\7~\2\2bc\7~\2\2c\22\3\2\2"+
		"\2de\7~\2\2e\24\3\2\2\2fg\7\60\2\2g\26\3\2\2\2hi\7=\2\2i\30\3\2\2\2jn"+
		"\t\3\2\2kn\5\23\n\2ln\t\4\2\2mj\3\2\2\2mk\3\2\2\2ml\3\2\2\2no\3\2\2\2"+
		"om\3\2\2\2op\3\2\2\2p\32\3\2\2\2qr\7>\2\2r\34\3\2\2\2st\7@\2\2t\36\3\2"+
		"\2\2uv\7/\2\2v \3\2\2\2wx\7p\2\2xy\7k\2\2y\u008d\7n\2\2z{\7v\2\2{|\7t"+
		"\2\2|}\7w\2\2}\u008d\7g\2\2~\177\7h\2\2\177\u0080\7c\2\2\u0080\u0081\7"+
		"n\2\2\u0081\u0082\7u\2\2\u0082\u008d\7g\2\2\u0083\u0084\7u\2\2\u0084\u0085"+
		"\7g\2\2\u0085\u0086\7n\2\2\u0086\u008d\7h\2\2\u0087\u0088\7u\2\2\u0088"+
		"\u0089\7w\2\2\u0089\u008a\7r\2\2\u008a\u008b\7g\2\2\u008b\u008d\7t\2\2"+
		"\u008cw\3\2\2\2\u008cz\3\2\2\2\u008c~\3\2\2\2\u008c\u0083\3\2\2\2\u008c"+
		"\u0087\3\2\2\2\u008d\"\3\2\2\2\u008e\u0090\t\5\2\2\u008f\u008e\3\2\2\2"+
		"\u0090\u0091\3\2\2\2\u0091\u008f\3\2\2\2\u0091\u0092\3\2\2\2\u0092\u0096"+
		"\3\2\2\2\u0093\u0095\t\6\2\2\u0094\u0093\3\2\2\2\u0095\u0098\3\2\2\2\u0096"+
		"\u0094\3\2\2\2\u0096\u0097\3\2\2\2\u0097$\3\2\2\2\u0098\u0096\3\2\2\2"+
		"\u0099\u009a\7`\2\2\u009a&\3\2\2\2\u009b\u009c\7<\2\2\u009c(\3\2\2\2\u009d"+
		"\u009e\7<\2\2\u009e\u009f\7?\2\2\u009f*\3\2\2\2\u00a0\u00a1\7%\2\2\u00a1"+
		",\3\2\2\2\u00a2\u00a3\7&\2\2\u00a3.\3\2\2\2\u00a4\u00a5\7g\2\2\u00a5\60"+
		"\3\2\2\2\u00a6\u00a7\7\63\2\2\u00a7\u00a8\78\2\2\u00a8\u00a9\7t\2\2\u00a9"+
		"\62\3\2\2\2\u00aa\u00ab\7%\2\2\u00ab\u00ac\7*\2\2\u00ac\64\3\2\2\2\u00ad"+
		"\u00ae\7%\2\2\u00ae\u00af\7}\2\2\u00af\66\3\2\2\2\u00b0\u00b1\7\177\2"+
		"\2\u00b18\3\2\2\2\u00b2\u00b3\7}\2\2\u00b3:\3\2\2\2\u00b4\u00b5\t\7\2"+
		"\2\u00b5<\3\2\2\2\u00b6\u00b7\t\b\2\2\u00b7>\3\2\2\2\u00b8\u00b9\5#\22"+
		"\2\u00b9\u00ba\5\'\24\2\u00ba@\3\2\2\2\u00bb\u00bc\5\'\24\2\u00bc\u00bd"+
		"\5#\22\2\u00bdB\3\2\2\2\u00be\u00c1\5-\27\2\u00bf\u00c2\5=\37\2\u00c0"+
		"\u00c2\5-\27\2\u00c1\u00bf\3\2\2\2\u00c1\u00c0\3\2\2\2\u00c2D\3\2\2\2"+
		"\13\2KTmo\u008c\u0091\u0096\u00c1\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}