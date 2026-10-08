package com.twitter.botmaker.compiler;

import java.util.Set;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;

import org.antlr.runtime.ANTLRStringStream;
import org.antlr.runtime.CharStream;
import org.antlr.runtime.CommonTokenStream;
import org.antlr.runtime.RecognitionException;
import org.antlr.runtime.tree.Tree;

import com.twitter.botmaker.antlr.BotMakerLexer;
import com.twitter.botmaker.antlr.BotMakerParser;
import com.twitter.botmaker.compiler.exceptions.ParseFailure;
import com.twitter.botmaker.compiler.tuples.Tuple;
import com.twitter.botmaker.compiler.tuples.Tuple3;

public final class Parser {

  private Parser() {
  }

  public static Tree createParseTree(String s) throws ParseFailure {
    CharStream input = new ANTLRStringStream(s);
    BotMakerLexer lexer = new BotMakerLexer(input);
    CommonTokenStream tokenStream = new CommonTokenStream(lexer);
    BotMakerParser parser = new BotMakerParser(tokenStream);
    BotMakerParser.parse_return root;
    try {
      root = parser.parse();
    } catch (Exception e) {
      String extraMsg = "";
      if (e.getCause() instanceof RecognitionException) {
        RecognitionException re = (RecognitionException) e.getCause();
        if (re.token != null) {
          String unexpectedTok = "<EOF>";
          if (re.getUnexpectedType() >= 0) {
            unexpectedTok = BotMakerParser.tokenNames[re.getUnexpectedType()];
          }
          extraMsg = String.format(
              ": line %d col %d, near token <%s>: unexpected token %s",
              re.token.getLine(), re.token.getCharPositionInLine() + 1,
              re.token.getText(), unexpectedTok
          );
        } else if (re.line != 0) {
          extraMsg = String.format(": line %d col %d", re.line, re.charPositionInLine + 1);
        }
      }
      throw new ParseFailure(String.format("Failed to parse string '%s'%s", s, extraMsg), e);
    }
    return (Tree) root.getTree();
  }

  public static Tuple3<Tree, ImmutableList<Tuple3<Tree, String, Optional<Tree>>>, Optional<Tree>>
  createParseParamsTree(String s) throws ParseFailure {

    ImmutableList<Tuple3<Tree, String, Optional<Tree>>> params = parseParams(s);
    Optional<Tree> returnType = Optional.absent();

    CharStream input = new ANTLRStringStream(s);
    BotMakerLexer lexer = new BotMakerLexer(input);
    CommonTokenStream tokenStream = new CommonTokenStream(lexer);
    BotMakerParser parser = new BotMakerParser(tokenStream);

    try {
      Tree root = (Tree) parser.parse_params().getTree();
      if (root.getChildCount() == 1) {
        Tree exprTree = root.getChild(0);
        return Tuple.of(exprTree, params, returnType);
      } else {
        Tree paramsTree = root.getChild(0);
        Tree exprTree = root.getChild(1);
        params = buildParams(paramsTree);
        if (paramsTree.getChild(0).getType() == BotMakerLexer.TYPE) {
          returnType = Optional.of(paramsTree.getChild(0));
        }
        return Tuple.of(exprTree, params, returnType);
      }
    } catch (Exception e) {
      String extraMsg = "";
      if (e.getCause() instanceof RecognitionException) {
        RecognitionException re = (RecognitionException) e.getCause();
        if (re.token != null) {
          String unexpectedTok = "<EOF>";
          if (re.getUnexpectedType() >= 0) {
            unexpectedTok = BotMakerParser.tokenNames[re.getUnexpectedType()];
          }
          extraMsg = String.format(
              ": line %d col %d, near token <%s>: unexpected token %s",
              re.token.getLine(), re.token.getCharPositionInLine() + 1,
              re.token.getText(), unexpectedTok
          );
        } else if (re.line != 0) {
          extraMsg = String.format(": line %d col %d", re.line, re.charPositionInLine + 1);
        }
      }
      throw new ParseFailure(String.format("Failed to parse string '%s'%s", s, extraMsg), e);
    }
  }

  public static boolean isLegacyDfSig(String expr) {
    if (!expr.startsWith("#param(") && !expr.startsWith("#params(")) {
      return false;
    }
    String sig = expr.substring(1);
    try {
      CharStream input = new ANTLRStringStream(sig);
      BotMakerLexer lexer = new BotMakerLexer(input);
      CommonTokenStream tokenStream = new CommonTokenStream(lexer);
      BotMakerParser parser = new BotMakerParser(tokenStream);
      Tree tree = (Tree) parser.params().getTree();
      return tree.getType() == BotMakerLexer.PARAMS;
    } catch (RecognitionException e) {
      return false;
    }
  }

  private static ImmutableList<Tuple3<Tree, String, Optional<Tree>>> parseParams(
      String s) throws ParseFailure {
    String expr;
    if (!s.startsWith("#param(") && !s.startsWith("#params(")) {
      return ImmutableList.of();
    } else {
      int nlIdx = s.indexOf('\n');
      if (nlIdx < 0) {
        nlIdx = s.length();
      }
      expr = s.substring(1, nlIdx);
    }

    CharStream input = new ANTLRStringStream(expr);
    BotMakerLexer lexer = new BotMakerLexer(input);
    CommonTokenStream tokenStream = new CommonTokenStream(lexer);
    BotMakerParser parser = new BotMakerParser(tokenStream);
    Tree root;
    try {
      root = (Tree) parser.params().getTree();
    } catch (Exception e) {
      if (e.getCause() instanceof RecognitionException) {
        RecognitionException re = (RecognitionException) e.getCause();
        String extraMsg = "";
        if (re.token != null) {
          String unexpectedTok = "<EOF>";
          if (re.getUnexpectedType() >= 0) {
            unexpectedTok = BotMakerParser.tokenNames[re.getUnexpectedType()];
          }
          extraMsg = String.format(
              ": line %d col %d, near token <%s>: unexpected token %s",
              re.token.getLine(), re.token.getCharPositionInLine() + 1,
              re.token.getText(), unexpectedTok);
        } else if (re.line != 0) {
          extraMsg = String.format(": line %d col %d", re.line, re.charPositionInLine + 1);
        }
        throw new ParseFailure(String.format(
            "Failed to parse DF signature '%s'%s", s, extraMsg), re);
      }
      throw new ParseFailure("unknown parse failure", e);
    }

    return buildParams(root);
  }

  private static ImmutableList<Tuple3<Tree, String, Optional<Tree>>> buildParams(
      Tree params) throws ParseFailure {
    ImmutableList.Builder<Tuple3<Tree, String, Optional<Tree>>> ret = ImmutableList.builder();
    Set<String> argNames = Sets.newHashSet();
    boolean hasOpt = false;
    for (int i = 0; i < params.getChildCount(); i++) {
      Tree arg = params.getChild(i);
      if (arg.getType() == BotMakerLexer.ARG) {
        String name = arg.getChild(1).getText();
        if (arg.getChildCount() == 2 && hasOpt) {
          throw new ParseFailure(
              String.format(
                  "Non-optional parameter %s must be declared before optional parameters.",
                  name));
        } else if (arg.getChildCount() == 2) {
          ret.add(Tuple.of(arg.getChild(0), name, Optional.absent()));
        } else {
          hasOpt = true;
          ret.add(Tuple.of(arg.getChild(0), name, Optional.of(arg.getChild(2))));
        }

        if (argNames.contains(name)) {
          throw new ParseFailure("Duplicated argument name " + name);
        }
        argNames.add(name);
      }
    }
    return ret.build();
  }

  public static Tree parseType(String expr) throws ParseFailure {

    CharStream input = new ANTLRStringStream(expr);
    BotMakerLexer lexer = new BotMakerLexer(input);
    CommonTokenStream tokenStream = new CommonTokenStream(lexer);
    BotMakerParser parser = new BotMakerParser(tokenStream);
    Tree root;
    try {
      root = (Tree) parser.type().getTree();
    } catch (Exception e) {
      if (e.getCause() instanceof RecognitionException) {
        RecognitionException re = (RecognitionException) e.getCause();
        String extraMsg = "";
        if (re.token != null) {
          String unexpectedTok = "<EOF>";
          if (re.getUnexpectedType() >= 0) {
            unexpectedTok = BotMakerParser.tokenNames[re.getUnexpectedType()];
          }
          extraMsg = String.format(
              ": line %d col %d, near token <%s>: unexpected token %s",
              re.token.getLine(), re.token.getCharPositionInLine() + 1,
              re.token.getText(), unexpectedTok);
        } else if (re.line != 0) {
          extraMsg = String.format(": line %d col %d", re.line, re.charPositionInLine + 1);
        }
        throw new ParseFailure(
            String.format("Failed to parse Type signature '%s'%s", expr, extraMsg),
            re);
      }
      throw new ParseFailure("unknown parse failure", e);
    }

    return root;
  }


  public static String printTree(Tree tree) {
    if (tree.getChildCount() == 0) {
      return String.format("leaf: %s, type; %d", tree.getText(), tree.getType());
    }
    String s = String.format(
        "\nnode: %s, type: %s, numChildren:%d\n",
        tree.getText(), tree.getType(), tree.getChildCount());
    for (int i = 0; i < tree.getChildCount(); i++) {
      s += printTree(tree.getChild(i)) + "\n";
    }
    return s;
  }
}
